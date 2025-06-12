

import torch.nn as nn
import torch

import os, math, argparse, functools, torch, evaluate
from torch.utils.data import DataLoader
from torchvision.models import resnet18, vgg16, ResNet18_Weights, VGG16_Weights
from transformers import (
    AutoModelForImageClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    PreTrainedModel,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5ForSequenceClassification,
    LlamaModel,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)

from torch.optim.lr_scheduler import CosineAnnealingLR
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import re, uuid, random
import numpy as np
import datasets



from img_dataloader import *
from gen_utils import *
from glue_utils import *
from checkpoint_load_utils import *




Img_cls_configuration_map = {
    "model" : {
        "dataset": {
            "epoch": 20,
            "lr": 0.001,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "momentum": 0.9,   
            "batch_size": 128
        }
    },

    "vgg16" : {
        "cifar10": {
            "epoch": 30,
            "lr": 0.001,
            "optimizer": "SGD",    
            "weight_decay": 0.01,    
            "momentum": 0.9,   # momentum
            "batch_size": 32
        },
        "gtsrb": {
            "epoch": 25,
            "lr": 0.0001,
            "optimizer": "Adam",   
            "weight_decay": 0.01,    
            "momentum": 0.9,
            "batch_size": 64
        },
        "svhn": {
            "epoch": 40,
            "lr": 0.0005,
            "optimizer": "Adam",   
            "weight_decay": 0.01,    
            "momentum": 0.9,
            "batch_size": 128
        }
    },

    "resnet18" : {
        "cifar10": {
            "epoch": 20,
            "lr": 0.001,
            "optimizer": "AdamW",       # β₁ = 0.9, β₂ = 0.999   
            "weight_decay": 0.0001,   # weight decay
            "momentum": 0.9,
            "batch_size": 128
        },
        "gtsrb": {
            "epoch": 20,
            "lr": 0.001,    
            "weight_decay": 0.0001,
            "optimizer": "AdamW",     # β₁ = 0.9, β₂ = 0.999
            "weight_decay": 0.0001,   # weight decay
            "momentum": 0.9,
            "batch_size": 128
        },
        "svhn": {
            "epoch": 30,
            "lr": 0.001,
            "optimizer": "Adam",
            "weight_decay": 0.01,
            "momentum": 0.9,
            "batch_size": 256
        }
    },

    "vit" : {
        "cifar100": {
            "epoch": 50,           # check again
            "lr": 0.001,
            "optimizer": "AdamW",     # β₁ = 0.9, β₂ = 0.999
            "weight_decay": 0.03,   # weight decay
            "momentum": 0.9,
            "batch_size": 128       # 512
        },
        "flower": {
            "epoch": 50,
            "lr": 0.0002,    
            "optimizer": "AdamW",     # β₁ = 0.9, β₂ = 0.999
            "weight_decay": 0.75,   # weight decay
            "momentum": 0.9,
            "batch_size": 128       # 256
        },
        "food": {
            "epoch": 30,
            "lr": 0.0001,
            "optimizer": "AdamW",     # β₁ = 0.9, β₂ = 0.999
            "weight_decay": 0.01,
            "momentum": 0.9,
            "batch_size": 128       # 256
        }
    },

    "llm" : {
        "glue" : {
            "epoch": 10,
            "lr": 0.001,    # 1e-3
            "optimizer": "AdamW",     # β₁ = 0.9, β₂ = 0.999
            "weight_decay": 0.01,   # default weight decay
            "momentum": 0,      # default momentum
            "batch_size": 8
        }
    }

}



class LlamaForSequenceClassification(nn.Module):
    def __init__(self, pretrained_name: str, num_labels: int, problem_type: str):
        super().__init__()
        self.base = LlamaModel.from_pretrained(
            pretrained_name             # , torch_dtype=torch.float16           # , trust_remote_code=True
        )
        self.config       = self.base.config           # PEFT expects this
        self.hidden_size  = self.config.hidden_size
        self.classifier   = nn.Linear(self.hidden_size, num_labels)
        self.problem_type = problem_type
        self.num_labels   = num_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,                       # <── catch any extra args (inputs_embeds, etc.)
    ):
        # pass ALL kwargs straight to the base model
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = out.last_hidden_state[:, -1, :]        # last token
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if self.problem_type == "regression":
                loss = nn.MSELoss()(logits.squeeze(), labels.squeeze())
            else:
                loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels),
                                              labels.view(-1))

        return type("Out", (), {"loss": loss, "logits": logits})
        # return {"loss": loss, "logits": logits}




checkpoint_map = {
    "model_name" : "checkpoint", 
    "vit" : "google/vit-base-patch16-224", 
    "vit-meta" : "facebook/deit-base-patch16-224", 
    "BERT" : "bert-base-uncased", 
    "T5" : "t5-base", 
    "LLaMA" : "meta-llama/Llama-2-7b-hf", 
    "RoBERTa" : "roberta-base"
}





def get_tokenizer (model_name):
    if "llama" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_map[model_name], use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            else:   # add brand-new pad token and record its ID
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_map[model_name], use_fast=True)

    return tokenizer







def get_optimizer(model, model_name, dataset, ft_type, train_length):

    if model_name in ["T5", "BERT", "LLaMA", "RoBERTa"]:
        model_name = "llm"
        if dataset in ['rte', 'mrpc', 'stsb', 'cola']:
            Img_cls_configuration_map[model_name]["glue"]["epoch"] = 20
        else:
            Img_cls_configuration_map[model_name]["glue"]["epoch"] = 10
        
        dataset = "glue"

    # set optimizers
    optimizer = Img_cls_configuration_map[model_name][dataset]["optimizer"]
    # schedular = None    # not using schedular in our training method
    lr = Img_cls_configuration_map[model_name][dataset]["lr"]
    weight_decay = Img_cls_configuration_map[model_name][dataset]["weight_decay"]
    momentum = Img_cls_configuration_map[model_name][dataset]["momentum"]
    max_epochs = Img_cls_configuration_map[model_name][dataset]["epoch"]

    # optim = optim_init(model, ft_type, optimizer, lr)
    optim = optim_init(model, ft_type, optimizer, lr, momentum, weight_decay)

    scheduler = None    
    # if model_name in ["resnet18", "vgg16"]:
        # scheduler = CosineAnnealingLR(optim, T_max=max_epochs)   # epoch-level step

    if model_name == "llm":
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=int(0.1 * train_length * max_epochs),      # gamma = 0.1
            num_training_steps=train_length * max_epochs,
        )

    return optim, scheduler, max_epochs




def _is_hf_peft(model) -> bool:
    """Return True if model is a HuggingFace PreTrainedModel or PEFT wrapper."""
    return isinstance(model, (PreTrainedModel, PeftModel))



def get_model (model_name, task="none", out_class=0, load_local=False, ft_type="lora", logger=None, exp_info="None", device="cuad:0", train_length=0):       # ft_type=['lora', 'fft', 'qlora']

    if model_name in ["vit", "vit-meta"]:
        model_checkpoint = checkpoint_map[model_name]
        model = AutoModelForImageClassification.from_pretrained(model_checkpoint)
        num_ftrs = model.classifier.in_features
        if out_class != 0:
            model.classifier = replace_last_layer(model.classifier, num_ftrs, out_class, replace=True)
            model.num_classes = out_class
        
    elif model_name == "BERT":
        model_checkpoint = checkpoint_map[model_name]
        problem_type = "regression" if task == "stsb" else "single_label_classification"
        model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=NUM_LABELS[task], problem_type=problem_type)
        num_ftrs = model.config.hidden_size

    elif model_name == "RoBERTa":
        model_checkpoint = checkpoint_map[model_name]
        problem_type = "regression" if task == "stsb" else "single_label_classification"
        model = RobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=NUM_LABELS[task], problem_type=problem_type)
        num_ftrs = model.config.hidden_size

    elif model_name == "T5":
        model_checkpoint = checkpoint_map[model_name]
        problem_type = "regression" if task == "stsb" else "single_label_classification"
        model = T5ForSequenceClassification.from_pretrained(model_checkpoint, num_labels=NUM_LABELS[task], problem_type=problem_type)
        num_ftrs = model.config.d_model

    elif model_name == "LLaMA":
        model_checkpoint = checkpoint_map[model_name]
        problem_type = "regression" if task == "stsb" else "single_label_classification"
        model = LlamaForSequenceClassification(model_checkpoint, NUM_LABELS[task], problem_type)
        num_ftrs = model.hidden_size
     
    elif model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features   # fully-connected layer input size
        if out_class !=0:
            model.fc = nn.Linear(num_ftrs, out_class)

    elif model_name == "vgg16":
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        replace_last_three_Linear_layer = True

        if replace_last_three_Linear_layer:
            num_ftrs = model.classifier[0].in_features  # which is 25088
            if out_class != 0:
                model.classifier = nn.Sequential(
                    nn.Flatten(),  # Ensures input is flattened (should be already handled by VGG16)
                    nn.Linear(num_ftrs, out_class)
                )
        else:       # just replace the last classifier layer : faster, less param to train - but not suitable if dataset is not related to imagenet
            num_ftrs = model.classifier[-1].in_features  # last Linear layer input
            if out_class !=0:
                model.classifier[-1] = nn.Linear(num_ftrs, out_class)

    else:
        raise ValueError("Unsupported model for loading.")
    

    # set fine-tuning approach configuration
                    # LoRA
    if ft_type == 'lora':
        lora_model = configure_LoRA(model, model_name=model_name)
        print_trainable_parameters(lora_model)
        model = lora_model

    elif ft_type == 'qlora':   # qlora
        qlora_model = configure_qLoRA(model)        # Not completely implemented yet
        print_trainable_parameters(qlora_model)
        model = qlora_model
    

    optim, scheduler, max_epochs = get_optimizer(model, model_name, dataset = task, ft_type=ft_type, train_length=train_length)
    saved_epoch = 0     # initially no model saved

    if load_local:
        # return load_from_local_checkpoint(model_name, dataset=task, ft_type=ft_type)
        saved_epoch = load_checkpoint(model, logger, optim, scheduler, model_name, task, ft_type, exp_info, map_location=device)
        # max_epochs -= epochs


    # print("testing ... ... ... ")
    # if model_name=="resnet18" and _is_hf_peft(model):
    #     print("resnet18 model is also peft model")
    # else:
    #     print("No - resent18 is not PEFT")

    return model, optim, scheduler, saved_epoch, max_epochs, num_ftrs




def get_target_module(model_name):
    if model_name.lower().startswith("bert") or "roberta" in model_name.lower():
        targets = ["query", "value"]          # BERT/Roberta
    elif model_name.lower().startswith("t5"):
        targets = ["q", "v"]                  # T5 uses shorthand names
    elif "llama" in model_name.lower():
        targets = ["q_proj", "v_proj"]        # LLaMA / GPT-style
    elif model_name in ["vit", "vit-meta"]:
        targets = ["query", "value"]          # ViT attention blocks

    elif model_name == "resnet18":
        # For ResNet, common layers to target are the convolutional layers.
        # You might need to be more specific based on which blocks you want to target.
        # This targets 'conv1' and all 'conv' layers within the 'layer' blocks.
        targets = [
            "conv1",
            "layer1.0.conv1", "layer1.0.conv2",
            "layer1.1.conv1", "layer1.1.conv2",
            "layer2.0.conv1", "layer2.0.conv2", "layer2.0.downsample.0", # downsample also has a conv layer
            "layer2.1.conv1", "layer2.1.conv2",
            "layer3.0.conv1", "layer3.0.conv2", "layer3.0.downsample.0",
            "layer3.1.conv1", "layer3.1.conv2",
            "layer4.0.conv1", "layer4.0.conv2", "layer4.0.downsample.0",
            "layer4.1.conv1", "layer4.1.conv2",
        ]

    elif model_name == "vgg16":
        # For VGG, target the convolutional layers within the 'features' sequential module.
        # VGG's classifier is linear, so LoRA might not be the best fit there, but conv layers are good.
        targets = [f"features.{i}" for i in range(len(vgg16(weights=None).features)) if isinstance(vgg16(weights=None).features[i], nn.Conv2d)]

    else:
        raise ValueError(f"No LoRA target map for {model_name}")
    
    return targets






def configure_LoRA(model, model_name="vit", rank=16, dropout=0.1):
    # if model_name in ["resnet18", "vgg16"]:
    #     print(f"Cannot apply LoRA configuration on {model_name} model")
    #     return model

    kwargs = dict(
        r              = rank,
        lora_alpha     = rank,                     # α = r ⇒ scale = 1
        lora_dropout   = dropout,
        target_modules = get_target_module(model_name),
        bias           = "none",
        # modules_to_save= ["classifier"],
    )
    if model_name.lower() in ["t5", "bert", ]:
        kwargs["task_type"] = TaskType.SEQ_CLS
    
    if model_name == "resnet18":
        kwargs["modules_to_save"] = ["fc"]
    else:
        kwargs["modules_to_save"] = ["classifier"]

    config = LoraConfig(**kwargs)
    lora_model = get_peft_model(model, config)

    return lora_model




"""

def configure_LoRA(model, model_name="vit", rank=16, dropout=0.1):
    # Define target modules based on model type
    if model_name == "resnet18":
        target_modules = ["fc"]  # Final classification layer
        modules_to_save = ["fc"]
    elif model_name == "vgg16":
        target_modules = ["classifier.6"]  # Last linear layer
        modules_to_save = ["classifier.6"]
    else:
        target_modules = get_target_module(model_name)
        modules_to_save = ["classifier"]

    kwargs = dict(
        r=rank,
        lora_alpha=rank,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=modules_to_save,
    )

    # if model_name not in ["vit", "vit-meta", "resnet18", "vgg16"]:
    if model_name in ["BERT", "T5"]: 
        kwargs["task_type"] = TaskType.SEQ_CLS

    config = LoraConfig(**kwargs)
    lora_model = get_peft_model(model, config)

    return lora_model
"""

"""
    # old approach 
    if model_name in ["vit", "vit-meta"]:
        config = LoraConfig(
            r=rank,     # 16
            lora_alpha=16,
            target_modules= get_target_module(model_name),       # ["query", "value"],      # suitable for LoRA
            lora_dropout=dropout,       # 0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
    else:
        config = LoraConfig(
            task_type      = TaskType.SEQ_CLS,    # sequence-classification
            r              = rank,      # 16,                  # LoRA rank
            lora_alpha     = 16,                  # scaling (α = r keeps scale == 1)
            lora_dropout   = dropout,
            target_modules = get_target_module(model_name),              # ["q", "v"],          # T5 attention uses q,k,v,o ➜ q & v are common
            bias           = "none",
            modules_to_save= ["classifier"],      # keep head out of LoRA so it stays trainable
        )
        
    lora_model = get_peft_model(model, config)

    return lora_model

"""



def configure_qLoRA(model):
    print("Not implemented yet")
    pass    # ToDo 




"""
def configure_QLoRA(
    checkpoint: str,                 # HF model id or local path
    num_labels: int,                 # needed for classification heads
    rank: int = 16,
    alpha: int | None = None,        # if None → alpha = rank
    dropout: float = 0.05,
    target_modules=("query", "value"),
    compute_dtype=torch.bfloat16,    # bf16 as in the QLoRA paper
    device_map: str = "auto",        # spread across GPUs if available
):
    
    # Load `checkpoint` in 4-bit (NF4) and attach LoRA adapters.
    # Returns a train-ready model whose *only* trainable params are LoRA + head.
    
    # 1) 4-bit quantisation recipe (QLoRA, Table 2)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        quantization_config=bnb_cfg,
        device_map=device_map,
        trust_remote_code=True,
    )

    # 2) Cast LayerNorms to fp32 & enable gradient-checkpoint mask
    base = prepare_model_for_kbit_training(base)

    # 3) LoRA adapter on top of quantised weights
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha or rank,           # convention: α = r
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
        modules_to_save=["classifier"],     # keep head in full precision
    )

    qlora_model = get_peft_model(base, lora_cfg)
    print_trainable_parameters(qlora_model)   # optional helper

    return qlora_model

"""




def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )




def replace_last_layer(module, num_ftrs, num_class, replace, conv=False):
    if not replace: return module
    if conv:
        module = nn.Conv2d(num_ftrs, num_class, kernel_size=(1, 1), stride=(1, 1))
    else:
        module = nn.Linear(num_ftrs, num_class)
    return module




def optim_init(model, ft_type, optimizer, lr, momentum, weight_decay):
    if ft_type == 'fft':
        trainable_params = model.parameters()

    elif ft_type == 'lora':
        trainable_params = [param for name, param in model.named_parameters() if 'lora_A' in name or 'lora_B' in name]
        trainable_params.extend([param for name, param in model.named_parameters() if 'classifier' in name])

    elif ft_type == 'qlora':        # 4-bit base + LoRA adapters
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    elif ft_type == 'partial_ft':
        trainable_params = model.classifier.parameters()



    if optimizer == "Adam":
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)         # 1e-5)
    
    elif optimizer == "AdamW":
        return torch.optim.AdamW(trainable_params, lr=2e-5, weight_decay=weight_decay)        # 0.01 : defaule weight decay
    
    elif optimizer == "SGD":
        # return torch.optim.SGD(trainable_params, lr=lr)
        return torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)




