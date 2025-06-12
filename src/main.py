


import argparse
import re
import random
import numpy as np
from transformers import (
    AutoModelForImageClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModel,
    LlamaForCausalLM,
    T5ForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    T5ForConditionalGeneration,
)

from torchvision.transforms import transforms
from peft import get_peft_model, LoraConfig, TaskType

from logger_utils import get_configurable_logger
from img_dataloader import *
from glue_utils import *
from gen_utils import *
from FT_utils import *








def parser_set():
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--model_name', dest='model_name', type=str, default='resnet18', choices=["vit", "resnet18", "vgg16", "vit-meta", "BERT", "T5", "LLaMA"])
    parser.add_argument('--dataset', dest='dataset', type=str, default='cifar100', choices=['cifar100', 'food', 'flower'])
    parser.add_argument('--tasks', dest='tasks', type=str, default="cola,sst2,mrpc,qqp,mnli,qnli,rte,stsb", choices=["cifar100,food,flower", "cola,sst2,mrpc,qqp,mnli,qnli,rte,stsb"])
    parser.add_argument('--device', dest='device', type=str, default='cuda:0')
    parser.add_argument('--ft_type', dest='ft_type', type=str, default='lora', choices=['lora', 'fft', 'qlora'])
    parser.add_argument("--resume_finetuning", action="store_true", dest="resume_finetuning", default=False, help="load model from local checkpoint")
    
    # GLUE related
    parser.add_argument("--debug", dest='debug', type=bool, default=False)
    parser.add_argument("--epochs", dest='epochs', type=int, default=5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3) 
    parser.add_argument('--optimizer', dest='optimizer', default="AdamW", choices=["SGD", "Adam", "AdamW"])
    parser.add_argument("--seed", type=int, default=9595)      # 9595)

    # ViT related
    parser.add_argument('--image_size', dest='image_size', type=int, default=224)
    # parser.add_argument('--max_iter', dest='max_iter', type=int, default=5000)  # or use epoch instead
    parser.add_argument('--train_batch_size', dest='train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=128)
    parser.add_argument("--save_interval", type=int, default=5)

    args = parser.parse_args()
    # args.model_checkpoint = "google/vit-base-patch16-224" if args.model_name == 'vit' else  "facebook/deit-base-patch16-224"
    return args



def set_visible_cuda(device_str: str):
    """
    • If device_str is "cuda" or "cuda:N", keep only GPU N visible.
    • For "cpu", hide all GPUs (CUDA won’t be initialised).
    Call it *before* importing torch so that CUDA is initialised
    with the desired mask.
    """
    if device_str.startswith("cuda"):
        # "cuda"     → keep GPU 0
        # "cuda:3"   → keep GPU 3
        m = re.match(r"cuda(?::(\d+))?$", device_str)
        gpu_idx = m.group(1) if m else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    else:  # "cpu" or anything else
        os.environ["CUDA_VISIBLE_DEVICES"] = ""



def set_seed(seed=0):
    if seed == 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pass



def load_dataset_for_vit(image_size, dataset, train_batch_size, test_batch_size):
    # load dataset for vit
    size =(image_size, image_size)
    # mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    # normalize = transforms.Normalize(mean, std)
    normalize = transforms.Normalize(mean_map[dataset], std_map[dataset])
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            normalize,
        ])

    train_loader, test_loader, val_loader = loader_map[dataset](root_map[dataset], batch_size=train_batch_size, transform=transform)

    return train_loader, test_loader, val_loader




def img_cls_evaluate(model, loader, device, criterion):
    model.eval()
    tot_loss, tot_correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs  = inputs.to(device)
            labels  = labels.to(device)
            raw_out = model(inputs)         # # Hugging-Face models return an object with .logits
            outputs = raw_out.logits if hasattr(raw_out, "logits") else raw_out
            # outputs  = model(inputs).logits
            input_size = inputs.size(0)

            loss = criterion(outputs, labels)
            tot_loss    += loss.item() * input_size
            tot_correct += (outputs.argmax(1) == labels).sum().item()

    n = len(loader.dataset)
    return tot_loss / n, tot_correct / n     # loss, accuracy




def main():
    
    args = parser_set()
    # args.debug = True
    for key, value in args.__dict__.items():
        print("{:<20} {}".format(key, value))

    device=torch.device(args.device)
    set_seed(args.seed)
    # model_ext = "bin"       # for peft models
    # resume_finetuning = False
    
    component = "base_run"
    exp_name = f"{args.model_name}_no_FI_{component}_{args.ft_type}_experiment_log"          # args.model_name + "_" + args.model_name + "_" + args.model_name + "_" + 
    logger = get_configurable_logger(
        exp_name,
        console_levels=("INFO", "WARNING", "ERROR"),   # change as you like
        log_dir="logs",
    )


    # specify tasks according to model name
    if args.model_name in ["vit", "vit-meta", "resnet18", "vgg16"]:
        if args.model_name in ["vit", "vit-meta"]:
            args.tasks = "flower"  # "cifar100,food,flower"
        else:
            args.tasks = "cifar10,gtsrb,svhn"

        datasets = args.tasks.split(",")
        logger.info(f"*************************** Dataset : {datasets} *****************************")

        for dataset in datasets:

            train_batch_size = Img_cls_configuration_map[args.model_name][dataset]["batch_size"]
            train_loader, test_loader, val_loader = load_dataset_for_vit(args.image_size, dataset, train_batch_size, train_batch_size)

            model, optim, scheduler, saved_epoch, max_epoch, num_ftrs = get_model (
                args.model_name, 
                task=dataset, 
                out_class=num_class_map[dataset], 
                load_local=args.resume_finetuning, 
                ft_type=args.ft_type, 
                logger=logger,
                exp_info=f"no_FI_{args.ft_type}",
                device=args.device, 
                train_length=len(train_loader.dataset)
            )

            # if args.resume_finetuning:
            #     checkpoint_name = f"checkpoints/{args.model_name}_{dataset}.{model_ext}"

            # model, optim, _ = get_model(args.model_name, out_class=num_class_map[dataset], load_local=args.resume_finetuning, stored_checkpoint=checkpoint_name)
            # print_trainable_parameters(model)            

            model = model.to(device)

                    
            ################################################## ViT Fine-tuning ##################################################
            current_iter = 0
            current_epoch = saved_epoch + 1
            model.train()
            criterion = nn.CrossEntropyLoss()

            # max_epoch = Img_cls_configuration_map[args.model_name][dataset]["epoch"]

            # if resume_finetuning:                       # ToDO :: Check codes here
            #     max_epoch -= stored_epoch
            #     optim = stored_optimizer
            #     # optimizer = stored_optimizer
            #     # lr = stored_lr
            #     # weight_decay = stored_weight_decay
            #     # momentum = stored_momentum
            # else:
            #     optimizer = Img_cls_configuration_map[args.model_name][dataset]["optimizer"]
            #     # schedular = None    # not using schedular in our training method
            #     lr = Img_cls_configuration_map[args.model_name][dataset]["lr"]
            #     weight_decay = Img_cls_configuration_map[args.model_name][dataset]["weight_decay"]
            #     momentum = Img_cls_configuration_map[args.model_name][dataset]["momentum"]
            #     optim = optim_init(model, args.ft_type, optimizer, lr, momentum, weight_decay)
            #     # optim = optim_init(model, args.ft_type, optimizer, lr)
            
            while current_epoch <= max_epoch:
                # logger.info("*" * 100)
                # logger.info(f"Iter: {current_iter}/{args.max_iter} Epoch {current_epoch}")

                model.train()
                run_loss, run_correct = 0.0, 0


                for inputs, labels in train_loader:     # training loop
                    current_iter += 1
                    optim.zero_grad()

                    inputs  = inputs.to(args.device)
                    labels  = labels.to(args.device)
                    raw_out = model(inputs)         # # Hugging-Face models return an object with .logits
                    outputs = raw_out.logits if hasattr(raw_out, "logits") else raw_out
                    # outputs  = model(inputs).logits
                    input_size = inputs.size(0)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optim.step()

                    run_loss    += loss.item() * input_size
                    run_correct += (outputs.argmax(1) == labels).sum().item()

                # ---- epoch metrics on training --------------------------------
                train_loss = run_loss / len(train_loader.dataset)
                train_acc  = run_correct / len(train_loader.dataset)

                # ---- evaluate on validation set ------------------------------- 
                val_loss, val_acc = img_cls_evaluate(model, val_loader, args.device, criterion)

                logger.info(
                    f"[{args.model_name} - {dataset} ]  Epoch {current_epoch:<3} "
                    f"| Train loss: {train_loss:.4f}  acc: {train_acc*100:.2f}% "
                    f"| Val loss: {val_loss:.4f}  acc: {val_acc*100:.2f}%"
                )

                if (current_epoch % args.save_interval) == 0:
                    save_checkpoint(model, logger, optim, args.model_name, dataset, current_epoch, args.ft_type, f"no_FI_{args.ft_type}")
           
                current_epoch += 1             

            # evaluate performance
            criterion = nn.CrossEntropyLoss()
            val_loss, val_acc = img_cls_evaluate(model, test_loader, args.device, criterion)

            logger.info("################################ FInal Test Accuracy ################################")
            logger.info(f"[{args.model_name} - {dataset}]  | Test loss: {val_loss:.4f} Test acc: {val_acc*100:.2f}%")
            # logger.info("Epoch {:<5} ACC: {:.2f}% Loss: {:.2f}".format(current_epoch, epoch_acc * 100, epoch_loss))


    else:   # T5 model :: Use GLUE tasks
        args.tasks = "cola,sst2,mrpc,qqp,mnli,qnli,rte,stsb"
        for task in args.tasks.split(","):
            logger.info(task)

            processed_dataset = preprocess_GLUE_dataset(task)
            tokenizer  = get_tokenizer (args.model_name)         # AutoTokenizer.from_pretrained("t5-base")

            ds = processed_dataset.map(functools.partial(preprocess_glue_task, tokenizer, task, args.max_len),
                        batched=True, remove_columns=processed_dataset["train"].column_names)
            # collate = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
            collate = DataCollatorWithPadding(
                tokenizer,  # handles padding/truncation
                padding="longest",
                return_tensors="pt",
            )
            train_dl = DataLoader(ds["train"], shuffle=True, batch_size=args.batch_size, collate_fn=collate, pin_memory=True)
            test_dl  = DataLoader(ds["test"], shuffle=False, batch_size=args.batch_size, collate_fn=collate, pin_memory=True)
            val_dl   = DataLoader(ds["validation"], shuffle=False, batch_size=args.batch_size, collate_fn=collate, pin_memory=True)


            model, optim, scheduler, saved_epoch, max_epoch, num_ftrs = get_model (
                args.model_name, 
                task=task, 
                out_class=0, 
                load_local=args.resume_finetuning, 
                ft_type=args.ft_type,  
                logger=logger,
                device=args.device, 
                train_length=len(train_dl)
            )

            # model, optim, scheduler, _ = get_model(args.model_name, task, load_local=args.resume_finetuning, stored_checkpoint=checkpoint_name)
            # # LoRA
            # if args.ft_type == 'lora':
            #     lora_model = configure_LoRA(model, model_name=args.model_name)
            #     print_trainable_parameters(lora_model)
            #     model = lora_model

            # elif args.ft_type == 'qlora':
            #     qlora_model = configure_qLoRA(model)        # Not completely implemented yet
            #     print_trainable_parameters(qlora_model)
            #     model = qlora_model

            model = model.to(device)

            # use dedicated functions with modified codes
            # optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
            # scheduler = get_linear_schedule_with_warmup(
            #     optim,
            #     num_warmup_steps=int(0.1 * len(train_dl) * args.epochs),
            #     num_training_steps=len(train_dl) * args.epochs,
            # )

            # if task in ['rte', 'mrpc', 'stsb', 'cola']:
            #     args.epochs = 20
            # else:
            #     args.epochs = 10
            # # args.epochs = max_epoch
            current_epoch = saved_epoch + 1
            scores_test = []
            scores_val = []
            # for e in range(1, max_epoch+1):
            while current_epoch <= max_epoch:
                loss=glue_train_one_epoch(model, train_dl, optim, scheduler, device, debug=args.debug)
                score_val=eval_glue(model, val_dl, task, device, debug=args.debug) 
                score_test = eval_glue(model, test_dl, task, device, debug=args.debug)
                scores_test.append(score_test)
                scores_val.append(score_val)
                logger.info(f"[Model={args.model_name}][TASK={task}] epoch {current_epoch}/{max_epoch} | loss {loss:.3f} | metric val {score_val*100:.1f} | metric test {score_test*100:.1f}")
                
                if (current_epoch % args.save_interval) == 0:
                    save_checkpoint(model, logger, optim, args.model_name, task, current_epoch, args.ft_type, f"no_FI_{args.ft_type}", scheduler)
                
                current_epoch += 1
            max_test_score = np.max(scores_test)
            max_test_score_id = np.argmax(scores_test)
            valid_val_score = scores_val[max_test_score_id]
            logger.info(f"[Model={args.model_name}][TASK={task}] valid metric val: {valid_val_score*100:.1f} | metric test {max_test_score*100:.1f}")
        


if __name__ == '__main__':
    main()



