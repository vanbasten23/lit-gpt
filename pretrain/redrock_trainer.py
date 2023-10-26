from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.strategies import FSDPStrategy, XLAStrategy
import numpy as np
import torch
import torch.autograd.profiler
from torch.distributed.fsdp import ShardingStrategy
import torch.multiprocessing as mp
import torch.profiler as tprofiler
from torch.utils.data import DataLoader, IterableDataset
import nvtx

import logging

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.model import GPT, Block
from lit_gpt.speed_monitor import SpeedMonitorCallback, estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, step_csv_logger, num_parameters


mp.set_start_method("spawn", force=True)

import utilities.monitor_collectives

utilities.monitor_collectives.shunt_torch_communication()

save_interval = int(os.environ.get("SAVE_INTERVAL", 10000))
eval_interval = 10000
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
decay_lr = True
min_lr = 6e-5

hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}


class LightningGPTModule(L.LightningModule):

  def __init__(
      self,
      config: Config,
      micro_batch_size,
      prof: Optional[tprofiler.profile],
      gradient_accumulation_steps: int,
      max_iters: int,
      warmup_iters: int,
      fast_init: bool,
      trainer,
  ) -> None:
    super().__init__()
    self.config = config
    self.module: Optional[torch.nn.Module] = None
    self.measured_flops: Optional[int] = None
    self.nsys_profile_step_multiple = 5
    self.micro_batch_size = micro_batch_size
    self.prof = prof
    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.backward_nvtx_range = None
    self.max_iters = max_iters
    self.warmup_iters = warmup_iters
    self.fast_init = fast_init
    self.trainer = trainer

  def configure_model(self) -> None:
    import psutil
    import time
    print(self.trainer.global_rank, ' in configure_model', flush=True)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush=True)
    if self.fast_init:
      t = time.time()
      # init meta model
      with self.trainer.init_module(empty_init=True):
        self.module = GPT(self.config)
      
      # Iterate through each module in the model, replacing meta layers
      # with real layers after initializing them and their weights.
      # All layers not initialized below will be initialized later by FSDP
      for name, module in self.module.named_modules():
        state_dict = {}
        if isinstance(module, torch.nn.Linear):
          # define new layer on cuda so weight initialization is much faster
          with torch.device('cuda'):
            new_linear = torch.nn.Linear(
                module.in_features,
                module.out_features,
                bias=True if module.bias is not None else False
            )

          # initialize weights
          new_linear.apply(self.module._init_weights)
          
          # move new layer to cpu & prepare to load into model
          new_linear.to('cpu')
          state_dict[f"{name}.weight"] = new_linear.weight

          if module.bias is not None:
            state_dict[f"{name}.bias"] = new_linear.bias

        elif isinstance(module, torch.nn.Embedding):
          # define new layer on cuda so weight initialization is much faster
          with torch.device('cuda'):
            new_embedding = torch.nn.Embedding(
                module.weight.size()[0],
                module.weight.size()[1]
            )

          # initialize weights
          new_embedding.apply(self.module._init_weights)
            
          # move new layer to cpu & prepare to load into model
          new_embedding.to('cpu')
          state_dict[f"{name}.weight"] = new_embedding.weight

        # load new layer's weights & biases into model
        self.module.load_state_dict(state_dict, strict=False, assign=True)
    else:
      t = time.time()
      self.module = GPT(self.config)
      print(f'{self.trainer.global_rank} time to init model: {(time.time()-t):.02f}s', flush=True)
      t = time.time()
      self.module.apply(self.module._init_weights)
    print(f'{self.trainer.global_rank} time to init weights: {(time.time()-t):.02f}s', flush=True)
    print(self.trainer.global_rank, ' out configure_model', flush=True)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush=True)



  def configure_optimizers(self) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        self.module.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
    )

  def on_fit_start(self) -> None:
    trainer = self.trainer
    with torch.device("meta"):
      meta_model = GPT(self.module.config)
      trainer.print('model size: ', num_parameters(meta_model))
      # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
      # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
      # consider setting `self.measured_flops = estimated_flops` instead
      estimated_flops = estimate_flops(meta_model) * self.micro_batch_size
      print(
          f"Estimated TFLOPs: {estimated_flops * trainer.world_size / 1e12:.2f}"
      )
      x = torch.randint(
          0, 1, (self.micro_batch_size, meta_model.config.block_size)
      )
      self.measured_flops = measure_flops(meta_model, x)
      print(
          "Measured TFLOPs:"
          f" {self.measured_flops * trainer.world_size / 1e12:.2f}"
      )

  def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
    if not decay_lr:
      return
    # determine and set the learning rate for this iteration
    lr = get_lr(self.trainer.fit_loop.total_batch_idx, self.max_iters, self.warmup_iters)
    for optimizer in self.trainer.strategy.optimizers:
      for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    global_batch_idx = batch_idx / self.gradient_accumulation_steps
    if (
        global_batch_idx > 0
        and global_batch_idx % self.nsys_profile_step_multiple == 0
    ):
      self.print(f"Starting Nsys profiling")
      torch.cuda.cudart().cudaProfilerStart()

  def on_train_batch_end(
      self, outputs, batch: Any, batch_idx: int, unused: int = 0
  ) -> None:
    global_batch_idx = batch_idx // self.gradient_accumulation_steps
    global_batch_offset = batch_idx % self.gradient_accumulation_steps
    is_first_microbatch = global_batch_offset == 0
    is_last_microbatch = global_batch_offset == self.gradient_accumulation_steps - 1
    if self.prof and is_last_microbatch:
      self.prof.step()

    if (
        global_batch_idx > 1
        and global_batch_idx % self.nsys_profile_step_multiple == 0
        and is_last_microbatch
    ):
      self.print(f"Stopping Nsys profiling")
      torch.cuda.cudart().cudaProfilerStop()
    if is_last_microbatch:
      self.print(f"HEARTBEAT: {global_batch_idx=}, {batch_idx=}")
      self.print(
          f"Max memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB"
      )
      sys.stdout.flush()
      sys.stderr.flush()

  @nvtx.annotate(color='green')
  def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
    input_ids, targets = batch
    # TODO: I think this is the forward pass
    logits = self.module(input_ids)
    # TODO: I think this is the backwards pass
    loss = chunked_cross_entropy(logits, targets, chunk_size=0)
    self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
    return loss

  def on_before_backward(self, loss):
    self.backward_nvtx_range = nvtx.start_range(message="backward", color="red")

  def on_after_backward(self):
    if self.backward_nvtx_range:
      nvtx.end_range(self.backward_nvtx_range)

  @nvtx.annotate(color='orange')
  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
    optimizer.step(closure=optimizer_closure)

  def validation_step(self, batch: Any, batch_idx: int) -> None:
    input_ids, targets = batch
    logits = self.module(input_ids)
    loss = chunked_cross_entropy(logits, targets, chunk_size=0)
    self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


def main(
    devices: int = 1,
    precision: Optional[str] = None,
    tpu: bool = False,
    model_name: str = "redrock-175b",
    name: str = "redrock-fsdp",
    max_iters: int = 60000,
    warmup_iters: int = 2000,
    out_dir: str = None,
    data_dir: str = None,
    num_nodes: int = 1,
    batch_size: int = 512,
    micro_batch_size: int = 4,
    use_pt_profiler: bool = False,
    pt_profiler_wait: int = 1,
    pt_profiler_warmup: int = 2,
    pt_profiler_active: int = 2,
    pt_profiler_repeat: int = 5,
    debug: bool = False,
    deterministic: bool = False,
    fast_init: bool = False,
) -> None:
  if use_pt_profiler:
    cm = nullcontext()
  else:
    cm = torch.autograd.profiler.emit_nvtx()
  with cm:
    precision = precision or get_default_supported_precision(
        training=True, tpu=tpu
    )

    base_out_dir = Path(out_dir)
    logger_out_dir = base_out_dir / "csv_logger"
    checkpoint_out_dir = base_out_dir / "checkpoints"
    tprofiler_out_dir = base_out_dir / "tprofiler"
    data_dir = Path(data_dir)

    gradient_accumulation_steps = batch_size // micro_batch_size
    assert gradient_accumulation_steps > 0
    if devices > 1:
      if tpu:
        # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
        devices = "auto"
        strategy = XLAStrategy(sync_module_states=False)
      else:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            # state_dict_type="sharded",
            limit_all_gathers=True,
            cpu_offload=False,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )
    else:
      strategy = "auto"

    logger = step_csv_logger(
        str(logger_out_dir),
        name,
        cls=CSVLogger,
        flush_logs_every_n_steps=log_interval,
    )
    speed_monitor = SpeedMonitorCallback(
        length_fn=lambda batch: batch[0].size(1),
        batch_size=micro_batch_size,
        window_size=50,
        time_unit="seconds",
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_out_dir,
        every_n_train_steps=save_interval,
        save_top_k=-1,
        verbose=True,
        save_weights_only=True,
    )
    trainer = L.Trainer(
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=logger,
        callbacks=[speed_monitor, model_checkpoint],
        max_steps=max_iters,
        max_epochs=1,
        limit_val_batches=eval_iters,
        accumulate_grad_batches=gradient_accumulation_steps,
        log_every_n_steps=log_interval,
        val_check_interval=eval_interval,
        num_nodes=num_nodes,
        deterministic=deterministic,
    )

    if debug:
      logging.getLogger('lightning.pytorch.trainer.trainer').setLevel(logging.DEBUG)

    L.seed_everything(
        1337, workers=True
    )  # same seed for every process to init model (FSDP)

    trainer.print(hparams)

    if trainer.global_rank == 0:
      base_out_dir.mkdir(parents=True, exist_ok=True)
      checkpoint_out_dir.mkdir(parents=True, exist_ok=True)
      logger_out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)
    trainer.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    if use_pt_profiler:
      prof = tprofiler.profile(
          schedule=tprofiler.schedule(
              wait=pr_profiler_wait,
              warmup=pt_profiler_warmup,
              active=pt_profiler_active,
              repeat=pt_profiler_repeat,
          ),
          on_trace_ready=tprofiler.tensorboard_trace_handler(tprofiler_out_dir),
          record_shapes=True,
          with_stack=True,
      )
      prof.start()
    else:
      prof = None
    model = LightningGPTModule(
        config,
        micro_batch_size,
        prof,
        gradient_accumulation_steps,
        max_iters,
        warmup_iters,
        fast_init,
        trainer,
    )
    trainer.print(
        f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds."
    )

    train_data = Dataset(str(data_dir / "train.bin"), config.block_size)
    val_data = Dataset(str(data_dir / "val.bin"), config.block_size)
    train_dataloader = DataLoader(
        train_data, batch_size=micro_batch_size, num_workers=2
    )
    val_dataloader = DataLoader(
        val_data, batch_size=micro_batch_size, num_workers=2
    )

    t0 = time.perf_counter()
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="last")
    trainer.print(f"Training time: {(time.perf_counter()-t0):.2f}s")
    if trainer.strategy.root_device.type == "cuda":
      trainer.print(
          f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB"
      )
    if prof:
      prof.stop()


class Dataset(IterableDataset):

  def __init__(self, data_file: Path, block_size: int):
    super().__init__()
    self.data_file = data_file
    self.block_size = block_size

  def __iter__(self):
    data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
    while True:
      i = torch.randint(len(data) - self.block_size, (1,)).item()
      x = torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
      y = torch.from_numpy(
          (data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
      )
      yield x, y


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, lr_decay_iters, warmup_iters):
  # 1) linear warmup for warmup_iters steps
  if it < warmup_iters:
    return learning_rate * it / warmup_iters
  # 2) if it > lr_decay_iters, return min learning rate
  if it > lr_decay_iters:
    return min_lr
  # 3) in between, use cosine decay down to min learning rate
  decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
  return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
  # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
  # torch.backends.cuda.enable_flash_sdp(False)
  torch.set_float32_matmul_precision("high")

  from jsonargparse import CLI

  CLI(main)
