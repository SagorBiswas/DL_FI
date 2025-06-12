


import argparse
import re
import random
import itertools
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

from fault_injection_handler import *




def parser_set():
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--model_name', dest='model_name', type=str, default='T5', choices=["vit", "resnet18", "vgg16", "vit-meta", "BERT", "T5", "LLaMA"])
    parser.add_argument('--dataset', dest='dataset', type=str, default='cifar100', choices=['cifar100', 'food', 'flower'])
    parser.add_argument('--tasks', dest='tasks', type=str, default="cola,sst2,mrpc,qqp,mnli,qnli,rte,stsb", choices=["cifar100,food,flower", "cola,sst2,mrpc,qqp,mnli,qnli,rte,stsb"])
    parser.add_argument('--device', dest='device', type=str, default='cuda:0')
    parser.add_argument('--ft_type', dest='ft_type', type=str, default='fft', choices=['lora', 'fft', 'qlora'])
    parser.add_argument("--resume_finetuning", action="store_true", dest="resume_finetuning", default=False, help="load model from local checkpoint")
    
    # fault injection
    parser.add_argument('--fi_component', dest='fi_component', type=str, default='base_run', choices=['base_run', 'weights', 'gradients', 'optim_states'])
    parser.add_argument('--fi_flip_bit_count', dest='fi_flip_bit_count', type=int, default=1)         # 1, 10, 100
    parser.add_argument('--fi_flip_iteration_pct', dest='fi_flip_iteration_pct', type=int, default=50)         # 0, 50, 100    - %
    parser.add_argument('--fi_flip_direction', dest='fi_flip_direction', type=str, default="all", choices=["01", "10", "all", "none"])  
    parser.add_argument('--fi_flip_time', dest='fi_flip_time', type=str, default='forward', choices=['forward', 'backward'])    
    parser.add_argument('--fi_bit_position', dest='fi_bit_position', type=int, default=2)         # 0,1,6,7,8
    # parser.add_argument('--fi_flip_bit_count', dest='fi_flip_bit_count', type=int, default=1)         # 1, 10, 100

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





def main_run(args, bit_position=-1, flip_iteration_pct=-1, flip_time="none", component="none", flip_direction="none", flip_bit_count=1):
    
    # args = parser_set()
    # args.debug = True
    print("inputs in cmd ... =>")
    for key, value in args.__dict__.items():
        print("{:<20} {}".format(key, value))
    print("#"*30)

    device=torch.device(args.device)
    set_seed(args.seed)

    bit_position = bit_position if bit_position != -1 else args.fi_bit_position  # random.randint(0, 8)   # inclusive: 0 ≤ n ≤ 8
    flip_iteration_pct = flip_iteration_pct if flip_iteration_pct != -1 else args.fi_flip_iteration_pct
    flip_time = flip_time if flip_time != "none" else args.fi_flip_time
    flip_direction = flip_direction if flip_direction != "none" else args.fi_flip_direction
    component = component if component != "none" else args.fi_component
    flip_bit_count = flip_bit_count if flip_bit_count != -1 else args.fi_flip_bit_count
    
    # component = "base_run" if args.fi_component=="None" else args.fi_component
    exp_name = f"{args.model_name}_{flip_bit_count}_{flip_direction}_flips_at_pos_{bit_position}_on_{component}_at_{flip_time}_{flip_iteration_pct}%_iteration_experiment_log"          # args.model_name + "_" + args.model_name + "_" + args.model_name + "_" + 
    logger = get_configurable_logger(
        exp_name,
        console_levels=("INFO", "WARNING", "ERROR"),   # change as you like
        log_dir="logs",
    )

    logger.info(f"component = {component}")
    logger.info(f"running experiment with params =>")
    logger.info(f"bit_position = {bit_position}")
    logger.info(f"flip_iteration_pct = {flip_iteration_pct}")
    logger.info(f"flip_time = {flip_time}")
    logger.info(f"flip_direction = {flip_direction}")
    logger.info(f"flip_bit_count = {flip_bit_count}")
    print("#"*30)


    # specify tasks according to model name
    if args.model_name in ["vit", "vit-meta", "resnet18", "vgg16"]:
        if args.model_name in ["vit", "vit-meta"]:
            args.tasks = "cifar100" # ,food,flower"
        elif args.model_name=="resnet18":
            args.tasks = "cifar10"
        else:       # "vgg16"
            args.tasks =  "gtsrb"             # "cifar10,gtsrb,svhn"

        datasets = args.tasks.split(",")
        logger.info(f"*************************** Dataset : {datasets} *****************************")

        for dataset in datasets:

            train_batch_size = Img_cls_configuration_map[args.model_name][dataset]["batch_size"]
            train_loader, test_loader, val_loader = load_dataset_for_vit(args.image_size, dataset, train_batch_size, train_batch_size)

            exp_info = f"{flip_bit_count}_{flip_direction}_flips_at_pos_{bit_position}_on_{component}_at_{flip_time}_{flip_iteration_pct}%_progress"

            model, optim, scheduler, saved_epoch, max_epoch, num_ftrs = get_model (
                args.model_name, 
                task=dataset, 
                out_class=num_class_map[dataset], 
                load_local=args.resume_finetuning, 
                ft_type=args.ft_type, 
                logger=logger,
                exp_info=exp_info,
                device=args.device, 
                train_length=len(train_loader)
            )          

            model = model.to(device)


            def flip_bit_in_component_img_model(running_model):
                logger.info("*****************************Flipping bits **************************************")
                success = flip_components(running_model, logger, component, num_to_flip = flip_bit_count, ft_type=args.ft_type, flip_dir=flip_direction, bit_pos=bit_position) 
                logger.info(f"successfully flipped {success} bits")
                
                im_val_loss, im_val_acc = img_cls_evaluate(running_model, val_loader, args.device, criterion)
                logger.info(
                    f"[{args.model_name} - {dataset} ]  Epoch {current_epoch:<3} "
                    f"| immediate Val loss: {im_val_loss:.4f}  immediate val acc: {im_val_acc*100:.2f}%"
                )
                logger.info("flipping function end")

                    
            ################################################## ViT Fine-tuning ##################################################
            current_iter = 0
            current_epoch = saved_epoch
            model.train()
            criterion = nn.CrossEntropyLoss()

            # max_epoch = args.epochs              # or whatever value you're using
            current_iter = saved_epoch * len(train_loader)
            max_iter = max_epoch * len(train_loader)
            flipping_iter = 1 if flip_iteration_pct==0 else int(max_iter*flip_iteration_pct/100)
            logger.info(f"Max epoch is = {max_epoch} and saved epoch is = {saved_epoch}")
            logger.info(f"Max iteration is = {max_iter} and Flipping iteration is = {flipping_iter}")

            while current_iter < max_iter:
            # while current_epoch < max_epoch:
                # logger.info("*" * 100)
                # logger.info(f"Iter: {current_iter}/{max_iter} Epoch {current_epoch}")

                model.train()
                run_loss, run_correct = 0.0, 0

                for inputs, labels in train_loader:     # training loop
                    current_iter += 1
                    optim.zero_grad()
# **********************************************
                    inputs  = inputs.to(args.device)
                    labels  = labels.to(args.device)                
                    # flip the bits before forward propagation :: ### flipping here is better            
                    if flip_time=="forward" and current_iter==flipping_iter and component != "none":     # "None" means no Fault Injection
                        print("#####################################################Flipping bit function called")
                        flip_bit_in_component_img_model(model)
# **********************************************
                    raw_out = model(inputs)         # # Hugging-Face models return an object with .logits
                    outputs = raw_out.logits if hasattr(raw_out, "logits") else raw_out
                    # outputs  = model(inputs).logits
                    input_size = inputs.size(0)

                    loss = criterion(outputs, labels)
                    # flip the bits before backward propagation
                    if flip_time=="backward" and current_iter==flipping_iter and component != "none":     # "None" means no Fault Injection
                        print("#####################################################Flipping bit function called")
                        flip_bit_in_component_img_model(model)
# **********************************************
                    loss.backward()
                    optim.step()
# **********************************************
                    run_loss    += loss.item() * input_size
                    run_correct += (outputs.argmax(1) == labels).sum().item()

                # ---- epoch metrics on training --------------------------------
                train_loss = run_loss / len(train_loader.dataset)
                train_acc  = run_correct / len(train_loader.dataset)

                # ---- evaluate on validation set ------------------------------- 
                val_loss, val_acc = img_cls_evaluate(model, val_loader, args.device, criterion)

                current_epoch += 1    
                logger.info(
                    f"[{args.model_name} - {dataset} ]  Epoch {current_epoch:<3}  Current Iteration {current_iter}"
                    f"| Train loss: {train_loss:.4f}  acc: {train_acc*100:.2f}% "
                    f"| Val loss: {val_loss:.4f}  acc: {val_acc*100:.2f}%"
                )

                if (current_epoch % args.save_interval) == 0:
                    save_checkpoint(model, logger, optim, args.model_name, dataset, current_epoch, args.ft_type, exp_info)
                    
            # if flipping_epoch == max_epoch and args.fi_component != "None":     # flipping at inference level    
            #     _ = flip_components(model, logger, args.fi_component, num_to_flip = flip_bit_count, ft_type=args.ft_type, bit_pos=2)
            #     logger.info("test metrics after flipping :: immediate")
                
            # evaluate performance
            criterion = nn.CrossEntropyLoss()
            val_loss, val_acc = img_cls_evaluate(model, test_loader, args.device, criterion)

            logger.info("################################ FInal Test Accuracy ################################")
            logger.info(f"[{args.model_name} - {dataset}]  | Test loss: {val_loss:.4f} Test acc: {val_acc*100:.2f}%")
            # logger.info("Epoch {:<5} ACC: {:.2f}% Loss: {:.2f}".format(current_epoch, epoch_acc * 100, epoch_loss))


    else:   # T5 model :: Use GLUE tasks
        args.tasks = "sst2,mnli"         # "cola,sst2,mrpc,qqp,mnli,qnli,rte,stsb"
        for task in args.tasks.split(","):
            logger.info(task)

            exp_info = f"{flip_bit_count}_{flip_direction}_flips_at_pos_{bit_position}_on_{component}_at_{flip_time}_{flip_iteration_pct}%_progress"

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
                exp_info=exp_info,
                device=args.device, 
                train_length=len(train_dl)
            )

            model = model.to(device)


            def flip_bit_in_component_glue_model(running_model):
                logger.info("*****************************Flipping bits **************************************")
                success = flip_components(running_model, logger, component, num_to_flip = flip_bit_count, ft_type=args.ft_type, flip_dir=flip_direction, bit_pos=bit_position)
                logger.info(f"successfully flipped {success} bits")
                
                im_score_val = eval_glue(running_model, val_dl, task, device, debug=args.debug) 
                logger.info(f"[Model={args.model_name}][TASK={task}] flipped :: immediate | metric val {im_score_val*100:.1f}")
                im_score_test = eval_glue(running_model, test_dl, task, device, debug=args.debug)
                logger.info(f"[Model={args.model_name}][TASK={task}] flipped at immediate | metric test {im_score_test*100:.1f}")
                logger.info("flipping function end")

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
            current_epoch = saved_epoch         # 0 if no checkpoint loading
            current_iter = saved_epoch * len(train_dl)      # 0 if saved epoch=0
            scores_test = []
            scores_val = []

            
            # flipping_epoch = int(max_epoch*flip_iteration_pct/100)
            max_iter = max_epoch * len(train_dl)
            flipping_iter = 1 if flip_iteration_pct==0 else int(max_iter*flip_iteration_pct/100)
            logger.info(f"Max epoch is = {max_epoch} and saved epoch is = {saved_epoch}")
            logger.info(f"Max iteration is = {max_iter} and Flipping iteration is = {flipping_iter}")

            # for e in range(1, max_epoch+1):
            while current_iter < max_iter:
            # while current_epoch < max_epoch:
                # if current_epoch==flipping_epoch and args.fi_component != "None":     # means no Fault Injection
                #     _ = flip_components(model, logger, args.fi_component, num_to_flip = flip_bit_count, ft_type=args.ft_type, bit_pos=bit_position)
                #     im_score_val = eval_glue(model, val_dl, task, device, debug=args.debug) 
                #     logger.info(f"[Model={args.model_name}][TASK={task}] flipped :: immediate | metric val {im_score_val*100:.1f}")
                #     im_score_test = eval_glue(model, test_dl, task, device, debug=args.debug)
                #     logger.info(f"[Model={args.model_name}][TASK={task}] flipped at inference level | metric test {im_score_test*100:.1f}")

                # loss, itr_count =glue_train_one_epoch(model, train_dl, optim, scheduler, device, debug=args.debug)                #     logger.info(f"[Model={args.model_name}][TASK={task}] flipped :: immediate | metric test {im_score_test*100:.1f}")

                loss, itr_count = glue_train_one_epoch_FI(
                    model, train_dl, optim, scheduler, device,
                    current_iter=current_iter,
                    flip_time=flip_time,
                    flipping_iter=flipping_iter,
                    component=component,
                    flip_fn=flip_bit_in_component_glue_model,
                    debug=args.debug
                )
                score_val = eval_glue(model, val_dl, task, device, debug=args.debug) 
                score_test = eval_glue(model, test_dl, task, device, debug=args.debug)
                
                scores_test.append(score_test)
                scores_val.append(score_val)

                current_epoch += 1
                current_iter = itr_count
                logger.info(f"[Model={args.model_name}][TASK={task}] epoch {current_epoch}/{max_epoch} | loss {loss:.3f} | metric val {score_val*100:.1f} | metric test {score_test*100:.1f}")
                
                if (current_epoch % args.save_interval) == 0:
                    save_checkpoint(model, logger, optim, args.model_name, task, current_epoch, args.ft_type, exp_info, scheduler)

            
            # if flipping_epoch == max_epoch and args.fi_component != "None":     # flipping at inference level  
            #     _ = flip_components(model, logger, args.fi_component, num_to_flip = flip_bit_count, ft_type=args.ft_type, bit_pos=bit_position)
            #     score_val = eval_glue(model, val_dl, task, device, debug=args.debug) 
            #     logger.info(f"[Model={args.model_name}][TASK={task}] flipped at inference level | metric val {score_val*100:.1f}")
            #     score_test = eval_glue(model, test_dl, task, device, debug=args.debug)
            #     logger.info(f"[Model={args.model_name}][TASK={task}] flipped at inference level | metric test {score_test*100:.1f}")

            max_test_score = np.max(scores_test)
            max_test_score_id = np.argmax(scores_test)
            valid_val_score = scores_val[max_test_score_id]
            logger.info(f"[Model={args.model_name}][TASK={task}] valid metric val: {valid_val_score*100:.1f} | metric test {max_test_score*100:.1f}")
        




def generate_main_run_kwargs(components, bit_position_groups, flip_iterations_percent, flip_times, flip_directions):             # (args):

    # Build list of combinations
    kwargs_list = []
    for flip_dir in flip_directions:
        for bit_group in bit_position_groups:
            for _ in range(2):  # Randomly sample 2 positions per group (like your original code)
                bit_position = random.choice(bit_group)
                for component, flip_iteration_pct, flip_time in itertools.product(components, flip_iterations_percent, flip_times):
                    kwargs = {
                        # "args": args,
                        "bit_position": bit_position,
                        "component": component,
                        "flip_iteration_pct": flip_iteration_pct,
                        "flip_time": flip_time,
                        "flip_direction": flip_dir
                    }
                    kwargs_list.append(kwargs)
    
    return kwargs_list



def main():
    args = parser_set()

    # Parameter options
    bit_position_groups = [
        [0, 1, 6, 7, 8],              # Probable bit positions
        [i for i in range(32)]        # All bit positions
    ]
    components = ["weight", "gradients"]
    flip_iterations_percent = [50, 100]
    flip_times = ["forward", "backward"]
    flip_direction = ["01", "10"]
    kwargs_list = generate_main_run_kwargs(components, bit_position_groups, flip_iterations_percent, flip_times, flip_direction)
    
    for kwargs in kwargs_list:
        main_run(args, **kwargs)

"""

def flip_time_selector(args, bit_position=-1, flip_iteration_pct=-1, component="none"):
    flip_times = ["forward", "backward"]

    for flip_time in flip_times:
        main_run(args, bit_position=bit_position, flip_iteration_pct=flip_iteration_pct, flip_time=flip_time, component=component)



def flip_iteration_selector(args, bit_position=-1, component="none"):
    flip_iterations_percent = [50, 100]

    for flip_iteration_percent in flip_iterations_percent:
        flip_time_selector(args, bit_position=bit_position, flip_iteration_pct=flip_iteration_percent, component=component)




def component_selector(args, bit_position=-1):
    components = ["weight", "gradients"]

    for component in components:
        flip_iteration_selector(args, bit_position=bit_position, component=component)




def bit_pos_selector(args):
    bit_position_groups = []
    probable_bit_pos = [0,1,6,7,8]      # fixed manually :: thanks to Kunbei
    bit_position_groups.append(probable_bit_pos)

    all_bit_pos = [i for i in range(32)]
    bit_position_groups.append(all_bit_pos)

    for pos_group in bit_position_groups:
        for pos_var in range(2):

            bit_position = pos_group[random.randint(0, len(pos_group))]
            component_selector(args, bit_position=bit_position)





def main():
    args = parser_set()
    # the function flow ->: bit_position vars -> component variables -> Flip iteration vars -> Flip time vars
    bit_pos_selector(args)


"""


if __name__ == '__main__':
    main()












