from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


# def get_batch(
#     dataset: npt.NDArray, batch_size: int, context_length: int, device: str
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))
#     x = torch.stack([
#             torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
#             for i in starting_idxs
#     ])  # fmt: skip
#     y = torch.stack(
#         [
#             torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
#             for i in starting_idxs
#         ]
#     )  # fmt: skip
#     if "cuda" in device:
#         x = x.pin_memory().to(device, non_blocking=True)
#         y = y.pin_memory().to(device, non_blocking=True)
#     else:
#         x = x.to(device)
#         y = y.to(device)
#     return x, y

""" Replaced get_batch with mine poor_man_dataloader from assignment-1 and renamed (functions and parameters) to get_batch"""

def get_batch(dataset: npt.NDArray, batch_size:int, context_length:int,device:str | None):
  last_index = len(dataset) - context_length
  start_idx = np.random.randint(0,last_index,size=batch_size,dtype=np.int64)
  offsets = np.arange(context_length+1,dtype= np.int64)
  ids = start_idx[:,None] + offsets[None,:]
  batch_np = np.asarray(dataset[ids], order="C")          # (B, T+1), likely uint16, order="C" makes the row major contiguous layout in memory
  batch = torch.from_numpy(batch_np).long()            # CPU torch.long, long() in int64, which is usually input into the model

  fast_transfer = device is not None and "cuda" in device

  if fast_transfer:
      batch = batch.pin_memory()

  X = batch[:, :-1]
  Y = batch[:,  1:]


  if device is not None:
      X = X.to(device, non_blocking=fast_transfer)
      Y = Y.to(device, non_blocking=fast_transfer)
  return (X,Y)
