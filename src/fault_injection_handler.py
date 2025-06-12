


import torch
import torch.nn as nn
import random

from typing import List, Dict, Any, Tuple

from bit_flip_utils import *


# -------------------------------------------------------------
# utility: collect float‐like tensors from any nested structure
# -------------------------------------------------------------
def _collect_fp_tensors(
    container: Any,
    parent_name: str = "",
) -> List[Tuple[str, torch.Tensor]]:
    """Recursively pull out tensors with fp32/fp16/bf16 dtype."""
    floats = []
    if isinstance(container, torch.Tensor):
        if container.dtype in {torch.float32, torch.float16, torch.bfloat16}:
            floats.append((parent_name, container))
    elif isinstance(container, dict):
        for k, v in container.items():
            floats += _collect_fp_tensors(v, f"{parent_name}.{k}" if parent_name else str(k))
    elif isinstance(container, (list, tuple)):
        for i, v in enumerate(container):
            floats += _collect_fp_tensors(v, f"{parent_name}[{i}]")
    return floats





def find_tensor_name(flat_index: int, tensor_ranges: List[Tuple[str, int, int]]) -> str:
    for name, start, end in tensor_ranges:
        if start <= flat_index < end:
            return name
    return "Unknown"


"""
# example usage of find_tensor_name function

    count, flipped, indices, tensor_ranges = flip_gradients(model, logger, num_to_flip=10)
    for idx in indices:
        name = find_tensor_name(idx, tensor_ranges)
        print(f"Global index {idx} came from: {name}")
    
"""

# -------------------------------------------------------------
# 1)  flip bits inside OPTIMIZER STATE
# -------------------------------------------------------------
from typing import Tuple, List

def flip_optimizer_states(
    optimizer: torch.optim.Optimizer,
    logger,
    *,
    num_to_flip: int = 1,
    ft_type: str = "fft",     # "fft" | "lora" | "qlora"
    flip_dir: str = "all",    # "01" | "10" | "all" | "none"
    bit_pos: int = 2,
    seed: int | None = None,
) -> Tuple[int, List[str], List[int], List[Tuple[str, int, int]]]:
    
    """
    Flip bits in optimizer state tensors (e.g., Adam moments).

    Returns:
        - success_count (int): Number of successful flips
        - flipped_ids (List[str]): Human-readable flipped entries (e.g., "param_1.exp_avg[42]")
        - flipped_indices (List[int]): Global flat indices of flipped elements
        - tensor_ranges (List[Tuple[str, int, int]]): Ranges for reverse mapping
    """

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # 1. collect fp tensors -------------------------------------------------
    names, tensors = zip(*_collect_fp_tensors(optimizer.state))
    tensor_ranges = []
    offset = 0

    for name, t in zip(names, tensors):
        numel = t.numel()
        tensor_ranges.append((name, offset, offset + numel))
        offset += numel

    total_elems = offset
    chosen = set(random.sample(range(total_elems), k=num_to_flip))

    # 2. mutate in-place ----------------------------------------------------
    flipped = []
    flipped_indices = []
    success_count = 0
    cursor = 0

    for name, t in zip(names, tensors):
        flat = t.view(-1)
        end = cursor + flat.numel()
        local = [i - cursor for i in chosen if cursor <= i < end]

        for idx in local:
            global_idx = idx + cursor
            old = flat[idx]
            actual_bit_pos = random.randint(0, 8) if bit_pos < 0 else bit_pos
            success, new = flip_bits_at_position(old, actual_bit_pos, flip_direction=flip_dir)
            flat[idx] = torch.tensor(new, dtype=t.dtype, device=t.device)

            flipped.append(f"{name}[{idx}]")
            flipped_indices.append(global_idx)
            success_count += success

        cursor = end

    logger.info(f"[Bit-Flip] optimizer: flipped {len(flipped)} elements "
                f"({len(flipped)/total_elems*100:.4f} %)")

    return success_count, flipped, flipped_indices, tensor_ranges



# -------------------------------------------------------------
# 2)  flip bits inside GRADIENTS
# -------------------------------------------------------------
def flip_gradients(
    model: torch.nn.Module,
    logger,
    *,
    num_to_flip: int = 1,
    ft_type: str = "fft",      # "fft" | "lora" | "qlora"
    flip_dir: str = "all",     # "01" | "10" | "all" | "none"  
    bit_pos: int = 2,
    seed: int | None = None,
) -> Tuple[int, List[str], List[int], List[Tuple[str, int, int]]]:  # success_count, flipped_ids, flipped_indices, tensor_ranges

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    grads, names, tensor_ranges = [], [], []
    offset = 0

    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.dtype in {torch.float32, torch.float16, torch.bfloat16}:
            names.append(name + ".grad")
            grads.append(p.grad)

            numel = p.grad.numel()
            tensor_ranges.append((name + ".grad", offset, offset + numel))
            offset += numel

    total_elems = offset
    chosen      = set(random.sample(range(total_elems), k=num_to_flip))

    flipped = []
    flipped_indices = []
    success_count = 0
    cursor = 0

    for name, g in zip(names, grads):
        flat = g.view(-1)
        end  = cursor + flat.numel()
        local = [i - cursor for i in chosen if cursor <= i < end]

        for idx in local:
            global_idx = idx + cursor
            old = flat[idx]
            if bit_pos < 0:
                bit_pos = random.randint(0, 8)
            success, new = flip_bits_at_position(old, bit_pos, flip_direction=flip_dir)
            flat[idx] = torch.tensor(new, dtype=g.dtype, device=g.device)

            flipped.append(f"{name}[{idx}]")
            flipped_indices.append(global_idx)
            success_count += success

        cursor = end

    logger.info(f"[Bit-Flip] gradients: flipped {len(flipped)} elements "
                f"({len(flipped)/total_elems*100:.4f} %)")

    return success_count, flipped, flipped_indices, tensor_ranges




def flip_model_weights(
    model,
    logger,
    *,
    num_to_flip: int = 1,
    ft_type: str = "fft",       # "fft" | "lora" | "qlora"
    flip_dir: str = "all",      # "01" | "10" | "all" | "none"
    bit_pos: int = 2,           # 2 ⇒ MSB of exponent
    flip_lora_only: bool = False,
    seed: int | None = None,
) -> Tuple[int, List[str], List[int], List[Tuple[str, int, int]]]:
    """
    Flip bits in model weights in-place.

    Returns:
        - success_count (int): Number of successful flips
        - flipped_ids (List[str]): List of "tensor_name[index]" strings
        - flipped_indices (List[int]): Global flat indices of flipped weights
        - tensor_ranges (List[Tuple[str, int, int]]): Offsets per tensor for reverse lookup
    """

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # 1. Collect candidate tensors and their ranges
    names, tensors, tensor_ranges = [], [], []
    offset = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ft_type == "lora" and flip_lora_only and not ("lora_" in name or "classifier" in name):
            continue

        names.append(name)
        tensors.append(p)

        numel = p.numel()
        tensor_ranges.append((name, offset, offset + numel))
        offset += numel

    total_elems = offset
    chosen = set(random.sample(range(total_elems), k=num_to_flip))

    # 2. Flip bits in-place
    flipped_ids = []
    flipped_indices = []
    success_count = 0
    start = 0

    for name, p in zip(names, tensors):
        flat = p.data.view(-1)
        end = start + flat.numel()
        local_ids = [i - start for i in chosen if start <= i < end]

        for idx in local_ids:
            global_idx = start + idx
            old_val = flat[idx]
            actual_bit_pos = random.randint(0, 8) if bit_pos < 0 else bit_pos
            success, new_val = flip_bits_at_position(old_val, actual_bit_pos, flip_direction=flip_dir)
            success_count += success
            flat[idx] = torch.tensor(new_val, dtype=p.dtype, device=p.device)

            flipped_ids.append(f"{name}[{idx}]")
            flipped_indices.append(global_idx)

        start = end

    logger.info(f"[Bit-Flip] corrupted {len(flipped_ids)} elements "
                f"({len(flipped_ids)/total_elems*100:.4f} %)")

    return success_count, flipped_ids, flipped_indices, tensor_ranges




def print_flipping_details(component, logger, success, indices, tensor_ranges):

    logger.info(f"Successfully flipped {success} bits from {component}")
    for idx in indices:
        name = find_tensor_name(idx, tensor_ranges)
        logger.info(f"Flipped at Global index {idx} which came from: {name}")
    logger.info(f"flipping details END")

    return
    




def flip_components(
    model,
    logger,
    component,
    # pct: float = 1,        # default flip 1% weights 
    *,
    num_to_flip: int = 1,
    ft_type: str = "fft",      # "fft" | "lora" | "qlora"
    flip_dir: str = "all",      # "01" | "10" | "all" | "none"  
    optimizer=None,
    bit_pos: int = 2,         # 2 ⇒ MSB of exponent
    # seed: int | None = None,
    seed: int = 42,                 # specifically mention seed=None, if you want no seed. bu default it's 42 - used for reproducibility
    write_flip_details = True
) :
    
    if component == "weights":
        success, flipped, indices, tensor_ranges = flip_model_weights(
            model,
            logger,
            num_to_flip=num_to_flip,
            ft_type=ft_type,   # "fft" for full models, "qlora" likewise
            flip_dir=flip_dir,
            bit_pos=bit_pos,       # MSB of exponent
            flip_lora_only=False,
            seed=seed                 # for reproducability
        )

        if write_flip_details:
            print_flipping_details(component, logger, success, indices, tensor_ranges)

        return success
    
    elif component == "gradients":
        success, flipped, indices, tensor_ranges = flip_gradients(
            model,
            logger,
            num_to_flip=num_to_flip,
            ft_type=ft_type,   # "fft" for full models, "qlora" likewise
            flip_dir=flip_dir,
            bit_pos=bit_pos,       # MSB of exponent
            seed=seed                 # for reproducability
        )

        if write_flip_details:
            print_flipping_details(component, logger, success, indices, tensor_ranges)

        return success

    elif component == "optim_states" and optimizer:
        success, flipped, indices, tensor_ranges = flip_optimizer_states(
            optimizer,
            logger,
            num_to_flip=num_to_flip,
            ft_type=ft_type,   # "fft" for full models, "qlora" likewise
            flip_dir=flip_dir,
            bit_pos=bit_pos,       # MSB of exponent
            seed=seed                 # for reproducability
        )
        
        if write_flip_details:
            print_flipping_details(component, logger, success, indices, tensor_ranges)

        return success
    
    else:
        return 0    








"""
# usages

# flip 0.01 % of trainable weights in a LoRA-fine-tuned BERT model
flipped = flip_model_weights(
    model,
    num_to_flip=1,
    ft_type="lora",   # "fft" for full models, "qlora" likewise
    bit_pos=2,       # MSB of exponent
    seed=42
)
print("Flipped tensors:", flipped[:10], "...")


# after building model / optimizer
flip_optimizer_states(optimizer, num_to_flip=1, bit_pos=2)

# post-backward, before optimizer.step()
flip_gradients(model, num_to_flip=1, bit_pos=2)


"""