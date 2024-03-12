import gc
import json

import math
import operator
import random
import time
import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import wraps
from pathlib import Path
from random import sample

import numpy as np
import pandas as pd
import pytorch_warmup as warmup
import scipy
import torch
from model import SPC_PLM
from task import Task, define_dataset_config, define_tasks_config
from torch import optim
from tqdm import tqdm
from utils import stream_redirect_tqdm

from sklearn.metrics import confusion_matrix, classification_report

device_count = torch.cuda.device_count()


def seed_everything(seed=2023):
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


if torch.cuda.is_available():
    print('Running on GPU')
    print('device_count: ', device_count)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print('Running on CPU')


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total parameters': total_num, 'Trainable parameters': trainable_num}


def split_n(chunk_length, sequence):
    if type(sequence) == dict:
        key_splits = {}
        for key, subseq in sequence.items():
            key_splits[key] = split_n(chunk_length, subseq)

        splits_count = len(next(iter(key_splits.values())))
        splits = []
        for i in range(splits_count):
            s = {}
            for key, subseq in key_splits.items():
                s[key] = subseq[i]
            splits.append(s)

        return splits

    else:
        splits = []
        splits_count = math.ceil(len(sequence) / chunk_length)
        for i in range(splits_count):
            splits.append(sequence[i * chunk_length:min(len(sequence), (i + 1) * chunk_length)])

        return splits


def retry_with_batchsize_halving(train_task=None):
    def inner(train_fn):
        @wraps(train_fn)
        def wrapper(*args, **kwargs):
            retry = True
            task = train_task or kwargs.get("task")
            input_data = kwargs["input_data"]
            batch_size = len(input_data)
            label = kwargs.get("label", [0] * batch_size)
            optimizer = kwargs["optimizer"]

            while retry and batch_size > 0:
                microbatches = split_n(batch_size, input_data)
                microlabels = split_n(batch_size, label)

                for microbatch, microlabel in zip(microbatches, microlabels):
                    try:
                        new_kwargs = dict(kwargs, input_data=microbatch, label=microlabel)
                        train_fn(*args, **new_kwargs)
                    except RuntimeError as e:
                        print(f"{e} Error in current task {task} with batch size {batch_size}. Retrying...")
                        batch_size //= 2
                        optimizer.zero_grad(set_to_none=True)
                        break
                    finally:
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    retry = False

            if retry:
                print(f"Skipping {task} batch... (size: {batch_size})")

        return wrapper

    return inner


@retry_with_batchsize_halving()
def train_minibatch(input_data, task, label, model, **kwargs):
    output, loss = model(input_data, label, task)[:2]
    train_losses[task.name].append(loss.item())
    loss.backward()
    del output


def main():
    parser = ArgumentParser()
    parser.add_argument("--status", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--from_checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument('--patience', type=int, default=5, help='early stop')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fine_tune_task", type=Task, nargs='+', default=list([Task.tweeteval_stance]), choices=list(Task),
                        required=False, help="claire, glue_sts_b, isear_v3, tweeteval_stance, list(['tweeteval_offensive','tweeteval_hate'])")
    parser.add_argument("--dataset_percentage", type=float, default=1.0)
    parser.add_argument("--optimizer_name", default="Adamax", type=str, help="Adam, Adamax, AdamW")
    parser.add_argument("--lr", type=float, default=5e-5, help="lr: 5e-5, 1e-5")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup_ratio: 0.1, 0.06")
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pooling_method", type=str, default="cls", help="pooling_method: cls, avg, cnn")
    parser.add_argument("--pretrained_model_path", type=str, default="./ptms/roberta-base", help="roberta-base")  # "bert-base-chinese"
    parser.add_argument("--output_hidden_states_flag", action="store_true", default=False, help="flag of output_hidden_states")
    parser.add_argument("--output_dir", type=str, default="./outputs/demo-v1/")

    parser.add_argument("--var_weight", type=float, default=0.0)
    parser.add_argument("--clu_weight", type=float, default=0.0)
    parser.add_argument("--normalize_flag", action="store_true", default=False, help="flag of normalize")
    parser.add_argument("--batch_sampling_flag", action="store_true", default=False, help="flag of sampling batch data")

    parser.add_argument("--task_type", type=str, default="cls", help="cls, res, multi")
    parser.add_argument("--pair_flag", action="store_true", default=False, help="flag of pair sentences")
    parser.add_argument("--tokenizer_add_e_flag", action="store_true", default=False, help="flag of add special_tokens e for claire")

    parser.add_argument("--save_model_flag", action="store_true", default=False, help="flag of save model checkpoint")
    parser.add_argument("--module_print_flag", action="store_true", default=True, help="flag of module structure print")

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)
    datasets_config = define_dataset_config(fine_tune_task=args.fine_tune_task, batch_size=args.bs, pair_flag=args.pair_flag)
    tasks_config = define_tasks_config(datasets_config, dataset_percentage=args.dataset_percentage)

    task_actions = []
    task2num = {}
    for task in iter(Task):
        if task not in datasets_config.keys():
            continue
        train_loader = tasks_config[task]["train_loader"]
        task_actions.extend([task] * len(train_loader))
        task2num[task.name] = len(train_loader)
    print("task2num: ", task2num)

    with open('./data/task2label.json', 'r', encoding='utf-8') as f:
        task2label = json.load(f)

    model = SPC_PLM(pretrained_model_path=args.pretrained_model_path,
                    tasks_config=tasks_config,
                    max_length=args.max_length,
                    dropout=args.dropout,
                    var_weight=args.var_weight,
                    clu_weight=args.clu_weight,
                    pooling_method=args.pooling_method,
                    task_type=args.task_type,
                    output_hidden_states=args.output_hidden_states_flag,
                    module_print_flag=args.module_print_flag,
                    normalize_flag=args.normalize_flag,
                    tokenizer_add_e_flag=args.tokenizer_add_e_flag)
    model.to(device)

    for name, param in model.named_parameters():
        print(name, ":", param.size(), param.requires_grad)
    print(get_parameter_number(model))

    params_opt = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer_name == "Adamax":
        optimizer = optim.Adamax(params=params_opt, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params_opt, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(params=params_opt, lr=args.lr, weight_decay=args.weight_decay)

    initial_epoch = 0
    training_start = int(time.time())

    num_epochs = args.epochs
    if args.from_checkpoint:
        print("Loading from checkpoint")
        checkpoint = torch.load(args.from_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1
        training_start = checkpoint["training_start"]
        warmup_scheduler = None
        lr_scheduler = None

    else:
        print("Starting training from scratch")
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=max(1, int(sum(task2num.values()) * num_epochs * args.warmup_ratio)))  # 10

    if args.from_checkpoint:
        num_epochs += initial_epoch - 1
    print(f"------------------ training-start:  {training_start} --------------------------)")

    print("333333\n")

    if args.from_checkpoint and args.status == "test":
        results_folder = Path(f"{args.output_dir.rstrip('/')}_{training_start}/test")
        evaluate_task(model, checkpoint['epoch'], tasks_config, datasets_config, results_folder, stream_redirect_tqdm(), eval_loader="test_loader",
                      task2label=task2label)
        print("test is finished| done")
        exit(0)

    loss_pre = None
    patience = 0
    for epoch in range(initial_epoch, num_epochs):
        print("epoch: ", epoch)
        if args.batch_sampling_flag:
            tasks_config = define_tasks_config(datasets_config, dataset_percentage=args.dataset_percentage)
            task_actions = []
            for task in iter(Task):
                if task not in datasets_config.keys():
                    continue
                train_loader = tasks_config[task]["train_loader"]
                task_actions.extend([task] * len(train_loader))

        with stream_redirect_tqdm() as orig_stdout:
            epoch_bar = tqdm(sample(task_actions, len(task_actions)), file=orig_stdout)
            model.train()

            global train_losses
            train_losses = defaultdict(list)
            for task_action in epoch_bar:
                tasks_config_t = tasks_config[task_action]
                if args.fine_tune_task is not None:
                    print("epoch: {}, task_action: {}".format(epoch, task_action))
                train_loader = tasks_config_t["train_loader"]
                epoch_bar.set_description(f"current task: {task_action.name} in epoch:{epoch}")

                data = next(iter(train_loader))
                optimizer.zero_grad(set_to_none=True)
                data_columns = [col for col in tasks_config_t["columns"] if col != "label"]

                input_data = list(zip(*(data[col] for col in data_columns)))
                if len(data_columns) == 1: input_data = list(map(operator.itemgetter(0), input_data))

                label = data["label"]

                if tasks_config_t['task_type'] == "res":
                    label = [e.to(torch.float32) if e.dtype == torch.float64 else e for e in label]
                    label = [e.to(device) for e in label]
                else:
                    if label.dtype == torch.float64: label = label.to(torch.float32)
                    label = label.to(device)

                train_minibatch(input_data=input_data,
                                task=task_action,
                                label=label,
                                model=model,
                                optimizer=optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                if warmup_scheduler:
                    lr_scheduler.step()
                    warmup_scheduler.dampen()

            train_losses = dict((k, sum(v) / len(v)) for k, v in train_losses.items())
            print("train loss: ", sum(train_losses.values()) / len(train_losses), train_losses)
            del train_losses

            results_folder = Path(f"{args.output_dir.rstrip('/')}_{training_start}")
            models_path = results_folder / "saved_models"
            models_path.mkdir(parents=True, exist_ok=True)

            eval_loss, _ = evaluate_task(model, epoch, tasks_config, datasets_config, results_folder, orig_stdout, eval_loader="dev_loader",
                                         task2label=task2label)
            _, test_result_df = evaluate_task(model, epoch, tasks_config, datasets_config, results_folder, orig_stdout, eval_loader="test_loader",
                                              task2label=task2label)

            if loss_pre is None or eval_loss < loss_pre:
                patience = 0
                loss_pre = eval_loss
                test_result_df.to_csv(str(results_folder / f"test_best_results.csv"), mode='w', index_label='Epoch')

                if args.save_model_flag:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_start': training_start
                    }, str(models_path / f'epoch.pkl'))

            else:
                patience += 1

            if patience >= args.patience:
                print('Early stoping. current patience: {}, current epoch: {} '.format(patience, epoch - patience))
                break
            else:
                print('Current patience: {}, current epoch: {} '.format(patience, epoch - patience))


def evaluate_task(model, epoch, tasks_config, datasets_config, results_folder, orig_stdout, eval_loader="dev_loader", task2label=None):
    model.eval()
    val_results = {}
    pred_results = {}
    task_columns = ["id", "sentence2", "sentence1", "label", "pred", "label_id", "pred_id"]

    with torch.no_grad():
        task_bar = tqdm([task for task in Task], file=orig_stdout)
        loss_items = defaultdict(list)
        for task in task_bar:
            if task not in datasets_config.keys(): continue
            print("epoch: {}, eval_loader: {}, task: {}".format(epoch, eval_loader, task))
            task_bar.set_description(task.name)
            task_config_t = tasks_config[task]
            val_loader = task_config_t[eval_loader]
            id2label = dict((k, v) for k, v in enumerate(task_config_t['class_names']))

            if task_config_t['task_type'] == "res":
                task_predicted_labels = [torch.empty(0, device=device)] * len(task_config_t['class_names'])
                task_labels = [torch.empty(0, device=device)] * len(task_config_t['class_names'])
            else:
                task_predicted_labels = torch.empty(0, device=device)
                task_labels = torch.empty(0, device=device)
            task_data = defaultdict(list)
            for val_data in val_loader:
                data_columns = [col for col in task_config_t["columns"] if col != "label"]
                input_data = list(zip(*(val_data[col] for col in data_columns)))

                if task_config_t['task_type'] == "res":
                    # For regression
                    label = val_data["label"]
                    label = [e.to(device) for e in label]
                    if len(data_columns) == 1:
                        input_data = list(map(operator.itemgetter(0), input_data))

                    model_output = model(input_data, label, task)
                    predicted_label = model_output[0]
                    loss_items[task.name].append(model_output[1].item())
                    for i in range(len(label)):
                        task_predicted_labels[i] = torch.hstack((task_predicted_labels[i], predicted_label[:, i].view(-1)))
                        task_labels[i] = torch.hstack((task_labels[i], label[i]))
                    for k in ["id", "sentence1", "sentence2"]:
                        task_data[k].extend(val_data[k])

                else:
                    # For classiciation
                    label = val_data["label"].to(device)
                    if len(data_columns) == 1:
                        input_data = list(map(operator.itemgetter(0), input_data))

                    model_output = model(input_data, label, task)

                    if task_config_t['num_classes'] > 1:
                        predicted_label = torch.argmax(model_output[0], -1)
                    else:
                        predicted_label = model_output[0]
                    loss_items[task.name].append(model_output[1].item())
                    task_predicted_labels = torch.hstack((task_predicted_labels, predicted_label.view(-1)))
                    task_labels = torch.hstack((task_labels, label))
                    for k in ["id", "sentence1", "sentence2"]: task_data[k].extend(val_data[k])

            metrics = datasets_config[task].metrics
            w_f1_result = None
            for metric in metrics:
                if metric.__name__ in ["f1_score"]:
                    metric_result = metric(task_labels.cpu(), task_predicted_labels.cpu(), average='macro')
                    w_f1_result = metric(task_labels.cpu(), task_predicted_labels.cpu(), average='weighted')
                elif metric.__name__ in ["accuracy_score"]:
                    metric_result = metric(task_labels.cpu(), task_predicted_labels.cpu())
                elif metric.__name__ in ["pearsonr", "spearmanr"]:
                    metric_result = []
                    for i in range(len(task_predicted_labels)):
                        per_metric_result = metric(task_labels[i].cpu(), task_predicted_labels[i].cpu())
                        if type(per_metric_result) == tuple or type(per_metric_result) == scipy.stats.stats.SpearmanrResult: per_metric_result = \
                            per_metric_result[0]
                        metric_result.append(per_metric_result)

                if type(metric_result) == tuple or type(metric_result) == scipy.stats.stats.SpearmanrResult:
                    metric_result = metric_result[0]

                if task.name in ["emobank"]:
                    for i, item in enumerate(["V", "A", "D"]):
                        key = task.name, metric.__name__ + f"_{item}"
                        val_results[key] = metric_result[i]
                        print(f"eval_results[{eval_loader}, {task.name}, {metric.__name__}_{item}] = {val_results[key]}")
                else:
                    val_results[task.name, metric.__name__] = metric_result
                    print(f"eval_results[{eval_loader}, {task.name}, {metric.__name__}] = {val_results[task.name, metric.__name__]}")

                if metric.__name__ in ["f1_score"]:
                    val_results[task.name, "w_" + metric.__name__] = w_f1_result
                    print(f"eval_results[{eval_loader}, {task.name}, w_{metric.__name__}] = {w_f1_result}")

            target_names = task2label[task.name]

            if task.name == 'tweeteval_sentiment':
                # Sentiment (Macro Recall)
                metric_ = "macro_recall"
                cls_report = classification_report(task_labels.cpu(), task_predicted_labels.cpu(), target_names=target_names, output_dict=True)
                val_results[task.name, metric_] = cls_report['macro avg']['recall']
                print(f"eval_results[{eval_loader}, {task.name}, {metric_}] = {val_results[task.name, metric_]}")
            elif task.name == 'tweeteval_stance':
                # Stance (Macro F1 of 'favor' and 'against' classes)
                metric_ = "af_f1"
                cls_report = classification_report(task_labels.cpu(), task_predicted_labels.cpu(), target_names=target_names, output_dict=True)
                f1_against = cls_report['against']['f1-score']
                f1_favor = cls_report['favor']['f1-score']
                val_results[task.name, metric_] = (f1_against + f1_favor) / 2
                print(f"eval_results[{eval_loader}, {task.name}, {metric_}] = {val_results[task.name, metric_]}")
            elif task.name == 'tweeteval_irony':
                # Irony (Irony class f1)
                metric_ = "irony_f1"
                cls_report = classification_report(task_labels.cpu(), task_predicted_labels.cpu(), target_names=target_names, output_dict=True)
                val_results[task.name, metric_] = cls_report['irony']['f1-score']
                print(f"eval_results[{eval_loader}, {task.name}, {metric_}] = {val_results[task.name, metric_]}")

            val_results[task.name, "loss"] = sum(loss_items[task.name]) / len(loss_items[task.name])
            print(f"eval_results[{eval_loader}, {task.name}, loss] = {val_results[task.name, 'loss']}")

            if eval_loader == "test_loader":
                if task_config_t['task_type'] == "res":
                    print("Res matrix: ")
                    print(val_results)
                else:
                    print("Fine-grained matrix: ")
                    print(classification_report(task_labels.cpu(), task_predicted_labels.cpu(), target_names=target_names, digits=4))

                    print("Confusion_matrix: ")
                    print(confusion_matrix(y_true=task_labels.cpu(), y_pred=task_predicted_labels.cpu(), normalize="true"))

            task_data["sentence2"] = task_data.pop("sentence2")
            task_data["sentence1"] = task_data.pop("sentence1")

            if task_config_t['task_type'] == "res":
                task_data["label_id"] = torch.stack(task_labels).permute(1, 0).cpu().detach().numpy().tolist()
                task_data["pred_id"] = torch.stack(task_predicted_labels).permute(1, 0).cpu().detach().numpy().tolist()
                task_data["label"] = list(task_data["label_id"])
                task_data["pred"] = list(task_data["pred_id"])
            else:
                task_data["label_id"] = task_labels.cpu().detach().numpy().astype(np.int16)
                task_data["pred_id"] = task_predicted_labels.cpu().detach().numpy().astype(np.int16)
                task_data["label"] = list(map(lambda e: id2label[e], task_data["label_id"]))
                task_data["pred"] = list(map(lambda e: id2label[e], task_data["pred_id"]))

            pred_results[task.name] = pd.DataFrame(data=task_data, columns=task_columns)

    result_df = pd.DataFrame(data=val_results, index=[epoch])
    result_df.to_csv(str(results_folder / f"{eval_loader.split('_')[0]}_results.csv"), mode='a', index_label='Epoch')

    pred_writer = pd.ExcelWriter(str(results_folder / f"{eval_loader.split('_')[0]}_predictions.xlsx"), engine="openpyxl")
    for e in pred_results.keys():
        pred_results[e].to_excel(pred_writer, sheet_name=e, index=None)
    pred_writer.save()

    return sum([sum(ll) / len(ll) for ll in loss_items.values()]) / len(loss_items), result_df  # instance-level val loss, data_frame


if __name__ == '__main__':
    main()
