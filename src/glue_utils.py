

import torch
import re, uuid, random
import numpy as np
import datasets
import evaluate


############################################################################# GLUE related functions ######################################################
# --------------------------------------------------------------------- metric helpers
metric_map = {
    "sst2": ["accuracy"],
    "mnli": ["accuracy"],
    "qnli": ["accuracy"],
    "rte": ["accuracy"],
    "cola": ["matthews_correlation"],
    "stsb": ["pearson", "spearmanr"],
    "mrpc": ["accuracy", "f1"],
    "qqp": ["accuracy", "f1"],
}

GLUE_KEYS = {
    # single-sentence tasks
    "cola":       ("sentence",  None),
    "sst2":       ("sentence",  None),

    # sentence-pair tasks that share the same names
    "mrpc":       ("sentence1", "sentence2"),
    "qqp":        ("question1", "question2"),
    "stsb":       ("sentence1", "sentence2"),
    "rte":        ("sentence1", "sentence2"),
    ###############     "rte":        ("premise", "hypothesis"),

    # tasks with their own naming
    "mnli":       ("premise",   "hypothesis"),
    "qnli":       ("question",  "sentence"),
    ###############     "wnli":       ("premise",   "hypothesis"),
}

NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
    "qnli": 2, "rte": 2,
    "mnli": 3,              # ← the missing one
    "stsb": 1,              # regression (MSE head)
}

def _load_metric(path: str, **kwargs):
    """
    Wrapper around `evaluate.load` that
    • forces RAM-only storage (`keep_in_memory=True`)
    • attaches a unique experiment_id to avoid cache-file conflicts
    """
    return evaluate.load(
        path,
        keep_in_memory=True,
        experiment_id=f"{path}-{uuid.uuid4()}",
        **kwargs,
    )





def build_metric(task: str):
    """
    Return a list of evaluate.Metric objects matching the GLUE task.
      • Accuracy-based tasks      → ['accuracy']
      • Regression (stsb)        → ['pearson', 'spearmanr']
      • Paraphrase (mrpc, qqp)   → ['accuracy', 'f1']
    """
    if task in {"sst2", "mnli", "qnli", "rte"}:
        return [_load_metric("glue", config_name=task)]

    if task == "cola":
        return [_load_metric("glue", config_name="cola")]

    if task == "stsb":  # regression → two separate correlations
        return [
            _load_metric("glue", config_name="stsb", module_type="metric", name="pearson"),
            _load_metric("glue", config_name="stsb", module_type="metric", name="spearmanr"),
        ]

    if task in {"mrpc", "qqp"}:  # paraphrase → accuracy + F1
        return [
            _load_metric("glue", config_name=task),  # accuracy
            _load_metric("f1"),                      # F1
        ]

    raise ValueError(f"Unknown GLUE task: {task}")





# --------------------------------------------------------------------- dataset prep
def preprocess_glue_task(tokenizer, task, max_len, examples):
    # ------------- pick the right columns -----------------------
    if task in ("sst2", "cola"):
        inputs = tokenizer(
            examples["sentence"],
            truncation=True, padding="max_length", max_length=max_len,
        )

    elif task in ("mrpc", "stsb", "rte"):
        inputs = tokenizer(
            examples["sentence1"], examples["sentence2"],
            truncation=True, padding="max_length", max_length=max_len,
        )

    elif task == "qqp":
        inputs = tokenizer(
            examples["question1"], examples["question2"],
            truncation=True, padding="max_length", max_length=max_len,
        )

    elif task == "mnli":
        inputs = tokenizer(
            examples["premise"], examples["hypothesis"],
            truncation=True, padding="max_length", max_length=max_len,
        )

    elif task == "qnli":
        inputs = tokenizer(
            examples["question"], examples["sentence"],
            truncation=True, padding="max_length", max_length=max_len,
        )

    else:
        raise ValueError(f"Unsupported GLUE task: {task}")

    # ------------- attach the label -----------------------------
    # Leave the datatype alone; it can be a scalar or list.
    inputs["labels"] = examples["label"]
    return inputs



def preprocess_GLUE_dataset(task, seed=0):
    # 1. Load the raw GLUE dataset
    ds = datasets.load_dataset("glue", "mnli" if task == "mnli" else task)

    # 2. Re-split exactly as described in the paper
    if task == "mnli":
        # Use mismatched as validation, matched as latency_test
        ds["validation"] = ds["validation_mismatched"]
        ds["test"] = ds["validation_matched"]

    elif task in {"rte", "mrpc", "stsb", "cola"}:  # datasets < 10 k samples
        val_split = ds["validation"].train_test_split(
            test_size=0.5, seed=seed, shuffle=True
        )
        ds["validation"] = val_split["train"]  # new validation
        ds["test"] = val_split["test"]  # new latency_test

    else:  # large datasets (sst2, qqp, qnli, …)
        old_val = ds["validation"]  # becomes latency_test
        tv_split = ds["train"].train_test_split(
            test_size=1000, seed=seed, shuffle=True
        )
        ds["train"] = tv_split["train"]
        ds["validation"] = tv_split["test"]  # 1 k-sample validation
        ds["test"] = old_val

    return ds





# --------------------------------------------------------------------- Fault injected train
def glue_train_one_epoch_FI(
    model, loader, optim, sched, device,
    current_iter, flip_time, flipping_iter,
    component, flip_fn, debug=False
):
    model.train()
    loss_sum = 0

    for batch in loader:
        current_iter += 1
        for k in batch:
            batch[k] = batch[k].to(device)

        if flip_time == "forward" and current_iter == flipping_iter and component != "none":
            print("#####################################################Flipping bit function called")
            flip_fn(model)

        out = model(**batch)
        loss = out["loss"] if isinstance(out, dict) else out.loss

        if flip_time == "backward" and current_iter == flipping_iter and component != "none":
            print("#####################################################Flipping bit function called")
            flip_fn(model)

        loss.backward()
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

        loss_sum += loss.item()
        if debug:
            break

    return loss_sum / len(loader), current_iter




# --------------------------------------------------------------------- train / eval
def glue_train_one_epoch(model, loader, optim, sched, device, debug=False):
    model.train(); loss_sum=0
    for batch in loader:
        for k in batch: batch[k] = batch[k].to(device)
        out = model(**batch)
        loss = out["loss"] if isinstance(out, dict) else out.loss
        loss.backward()
        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
        loss_sum += loss.item()
        if debug: break
    return loss_sum / len(loader)




def eval_glue(model, loader, task, device, debug=False):
    model.eval();ms = build_metric(task)
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            for k in batch: batch[k] = batch[k].to(device)
            out = model(**batch)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            preds = torch.argmax(logits, dim=-1)
            for m in ms:
                if task=="stsb":  # regression
                    m.add_batch(predictions=logits.squeeze().cpu(), references=labels)
                else:
                    m.add_batch(predictions=preds.cpu(), references=labels)
            if debug: break
    res = [m.compute() for m in ms]
    # combine MRPC/QQP : (acc+f1)/2
    if task in ["mrpc","qqp"]:
        return sum(r[list(r.keys())[0]] for r in res)/len(res)
    if task in ["sst2","mnli","qnli","rte"]:
        return res[0]["accuracy"]
    if task=="cola":
        return res[0]["matthews_correlation"]
    if task=="stsb":
        return (res[0]["pearson"] + res[1]["spearmanr"])/2
    




    
