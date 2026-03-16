# Copyright (c) 2024 RedAccel Authors. All Rights Reserved.

import contextlib
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import BatchEncoding
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)


@contextlib.contextmanager
def disable_deepspeed_zero3(training_arguments):
    if training_arguments.deepspeed and is_deepspeed_zero3_enabled():
        unset_hf_deepspeed_config()
        try:
            yield
        finally:
            set_hf_deepspeed_config(training_arguments.hf_deepspeed_config)
    else:
        yield


def get_global_max_seq_len(tensor: torch.Tensor) -> int:
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor.shape[1]

    length = torch.Tensor([tensor.shape[1]]).to(tensor.device)
    length_list = [torch.empty_like(length, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(length_list, length)
    max_len = torch.stack(length_list).to(torch.int64).max().item()

    return int(max_len)


def gather_inputs(inputs: BatchEncoding, max_len: int) -> BatchEncoding:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size == 1:
        return inputs

    gathered_data = {}

    for key, tensor in inputs.items():
        tensor = tensor[:, :max_len]
        tensor = F.pad(tensor, (0, max_len - tensor.shape[1]), "constant", 0)

        tensor_list = [torch.empty_like(tensor, device=tensor.device) for _ in range(world_size)]
        dist.gather(tensor, gather_list=tensor_list if rank == 0 else None)
        gathered_tensor = torch.cat(tensor_list)
        gathered_data[key] = gathered_tensor

    return BatchEncoding(gathered_data)


def scatter_outputs(logits: Optional[torch.Tensor], shape: List[int], device, src: int = 0) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    output_tensor = torch.empty(shape, device=device)

    if rank == 0:
        if logits is None:
            raise ValueError("scatter logits should not be None ")
        tensor_list = list(torch.split(logits, logits.shape[0] // world_size, dim=0))
    else:
        tensor_list = None
    dist.scatter(output_tensor, tensor_list, src=src)
    return output_tensor
