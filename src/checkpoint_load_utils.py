# checkpoint_load_utils.py
import torch, json
from pathlib import Path
from typing import Optional
from transformers import PreTrainedModel
from peft import PeftModel

from logger_utils import *

project_home = Path(__file__).resolve().parent.parent


def create_directory():
    # project_home = Path(__file__).resolve().parent  # adjust for your env
    fname = f"{project_home}/checkpoints"
    _CHECK_DIR = Path(fname)
    _CHECK_DIR.mkdir(exist_ok=True)

    return _CHECK_DIR



def _is_hf_peft(model) -> bool:
    """Return True if model is a HuggingFace PreTrainedModel or PEFT wrapper."""
    return isinstance(model, (PreTrainedModel, PeftModel))


def _checkpoint_path(model_name: str, dataset: str, ft_type: str, is_hf: bool, exp_info: str="None"):
    _CHECK_DIR = create_directory()
    ext = ".bin" if is_hf else ".pth"
    fname = f"{model_name}_{dataset}_{ft_type}_flipped_{exp_info}{ext}"
    return _CHECK_DIR / fname


def save_checkpoint(
    model,
    logger,
    optimizer,
    model_name: str,
    dataset: str,
    epoch: int,
    ft_type: str,
    exp_info: str="None",
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,   # ← optional,
):
    is_hf = _is_hf_peft(model)
    ckpt_file = _checkpoint_path(model_name, dataset, ft_type, is_hf, exp_info)

    ckpt = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }

    if is_hf and model_name not in ["resnet18", "vgg16"]:
        ckpt["state_dict"] = model.state_dict()          # adapter + head for PEFT
        ckpt["config_json"] = model.config.to_json_string()
    else:                                                 # CNN (torchvision)
        ckpt["state_dict"] = model.state_dict()

    torch.save(ckpt, ckpt_file)
    logger.info(f"[Checkpoint] saved  → {ckpt_file} after epoch {epoch}")




def load_checkpoint(model, logger, optimizer, scheduler, model_name, dataset, ft_type, exp_info="None", map_location="cpu"):
    
    is_hf = _is_hf_peft(model)
    ckpt_file = _checkpoint_path(model_name, dataset, ft_type, is_hf, exp_info)

    try:
        ckpt = torch.load(ckpt_file, map_location=map_location)
    except FileNotFoundError:
        logger.warning(f"[Checkpoint] file not found → {ckpt_file}. Starting from scratch.")
        return 0

    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt["scheduler"]:
        scheduler.load_state_dict(ckpt["scheduler"])

    logger.info(f"[Checkpoint] loaded ← {ckpt_file}")
    epoch = int(ckpt["epoch"])

    return epoch






"""
CNN mdoels Example usage
# --- training loop ---
for epoch in range(1, num_epochs+1):
    train(...)
    validate(...)
    save_checkpoint(model, optimizer, scheduler, model_name, dataset, epoch)

# --- resume later ---
start_epoch = load_checkpoint(model, optimizer, scheduler, model_name, dataset)
for epoch in range(start_epoch+1, num_epochs+1):


PEFT models Example Usage
for epoch in range(1, 6):
    train(...)
    save_checkpoint(model, optimizer, scheduler, model_name, dataset, epoch)

# reload latest
load_checkpoint(model, optimizer, scheduler, model_name, dataset)  # picks highest eXXX

"""