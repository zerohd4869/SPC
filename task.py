import os
from enum import Enum, unique

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset
from scipy.stats import pearsonr, spearmanr


@unique
class Task(Enum):
    # classification
    tweeteval_emoji = "tweeteval_emoji"
    tweeteval_emotion = "tweeteval_emotion"
    tweeteval_hate = "tweeteval_hate"
    tweeteval_irony = "tweeteval_irony"
    tweeteval_offensive = "tweeteval_offensive"
    tweeteval_sentiment = "tweeteval_sentiment"
    tweeteval_stance = "tweeteval_stance"
    isear_v3 = "isear_v3"
    meld = "meld"
    goemotions = "goemotions"

    # regression
    glue_sts_b = "glue_sts_b"
    claire_v2 = "claire_v2"
    emobank = "emobank"

    def num_classes(self, tasks_config):
        return tasks_config[self]['num_classes']


class TaskConfig:
    def __init__(self, dataset_script, columns, batch_size, eval_batch_size, metrics, task_type="cls"):
        self.dataset_script = dataset_script
        self.columns = columns
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.metrics = metrics
        self.task_type = task_type


def define_dataset_config(fine_tune_task=None, batch_size=64, pair_flag=False):
    task_columns = ["label", "sentence1", "sentence2"] if pair_flag else ["label", "sentence1"]
    datasets_config = {
        # classification
        Task.isear_v3: TaskConfig(dataset_script="./datasets/isear_v3.py", columns=task_columns,
                                  batch_size=batch_size, eval_batch_size=batch_size * 2, metrics=[accuracy_score, f1_score]),
        Task.goemotions: TaskConfig(dataset_script="./datasets/goemotions.py", columns=task_columns,
                                    batch_size=batch_size, eval_batch_size=batch_size * 2, metrics=[accuracy_score, f1_score]),
        Task.meld: TaskConfig(dataset_script="./datasets/meld.py", columns=task_columns,
                              batch_size=batch_size, eval_batch_size=batch_size * 2, metrics=[accuracy_score, f1_score]),
        Task.tweeteval_emoji: TaskConfig(dataset_script="./datasets/tweeteval_emoji.py", columns=task_columns,
                                         batch_size=batch_size, eval_batch_size=batch_size * 2, metrics=[accuracy_score, f1_score]),
        Task.tweeteval_emotion: TaskConfig(dataset_script="./datasets/tweeteval_emotion.py", columns=task_columns,
                                           batch_size=batch_size, eval_batch_size=batch_size * 2,
                                           metrics=[accuracy_score, f1_score]),
        Task.tweeteval_hate: TaskConfig(dataset_script="./datasets/tweeteval_hate.py", columns=task_columns,
                                        batch_size=batch_size, eval_batch_size=batch_size * 2,
                                        metrics=[accuracy_score, f1_score]),
        Task.tweeteval_irony: TaskConfig(dataset_script="./datasets/tweeteval_irony.py", columns=task_columns,
                                         batch_size=batch_size, eval_batch_size=batch_size * 2,
                                         metrics=[accuracy_score, f1_score]),
        Task.tweeteval_offensive: TaskConfig(dataset_script="./datasets/tweeteval_offensive.py", columns=task_columns,
                                             batch_size=batch_size, eval_batch_size=batch_size * 2,
                                             metrics=[accuracy_score, f1_score]),
        Task.tweeteval_sentiment: TaskConfig(dataset_script="./datasets/tweeteval_sentiment.py", columns=task_columns,
                                             batch_size=batch_size, eval_batch_size=batch_size * 2,
                                             metrics=[accuracy_score, f1_score]),
        Task.tweeteval_stance: TaskConfig(dataset_script="./datasets/tweeteval_stance.py", columns=task_columns,
                                          batch_size=batch_size, eval_batch_size=batch_size * 2,
                                          metrics=[accuracy_score, f1_score]),
        # regression
        Task.glue_sts_b: TaskConfig(dataset_script="./datasets/glue_sts_b.py", columns=task_columns,
                                    batch_size=batch_size, eval_batch_size=batch_size * 2,
                                    metrics=[pearsonr, spearmanr], task_type="res"),
        Task.claire_v2: TaskConfig(dataset_script="./datasets/claire_v2.py", columns=task_columns,
                                   batch_size=batch_size, eval_batch_size=batch_size * 2,
                                   metrics=[pearsonr, spearmanr], task_type="res"),
        Task.emobank: TaskConfig(dataset_script="./datasets/emobank.py", columns=task_columns,
                                 batch_size=batch_size, eval_batch_size=batch_size * 2,
                                 metrics=[pearsonr, spearmanr], task_type="res"),

    }
    if fine_tune_task is not None:
        if type(fine_tune_task) == Task and fine_tune_task in datasets_config.keys():
            # single task
            datasets_config = dict((k, v) for k, v in datasets_config.items() if k == fine_tune_task)
        else:
            datasets_config = dict((k, v) for k, v in datasets_config.items() if k in fine_tune_task)

    print("## datasets_config: ", datasets_config)
    print("## sentences_pair_flag: ", pair_flag)
    print("## fine_tune_task: ", fine_tune_task)
    return datasets_config


def define_tasks_config(datasets_config, dataset_percentage=1.0, cache_dir="./datasets/cache/", class_weights_flag=False, seed=42):
    tasks_config = {}
    for id, (task, task_config) in enumerate(datasets_config.items()):
        print("task: ", task)
        print("task_config: ", task_config)

        if not os.path.isdir(cache_dir): os.makedirs(cache_dir)
        dataset_dic = load_dataset(path=task_config.dataset_script, cache_dir=cache_dir, split=None)
        train_dataset, val_dataset, test_dataset = dataset_dic["train"], dataset_dic["validation"], dataset_dic["test"]
        len_dataset = len(train_dataset)
        print("## dataset_dic.shape: ", dataset_dic.shape)
        if dataset_percentage < 1: np.random.seed(seed)
        train_dataset = train_dataset.select(
            list(np.random.choice(np.arange(len_dataset), int(len_dataset * dataset_percentage if dataset_percentage <= 1 else dataset_percentage), False)))
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=task_config.batch_size, shuffle=len_dataset > 0)
        dev_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=task_config.eval_batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=task_config.eval_batch_size, shuffle=False)

        if class_weights_flag:
            import sys, importlib
            sys.path.append("./datasets")
            dataset_module = importlib.import_module(task.name)
            formatted_name = "".join([w.capitalize() for w in task.name.split('_')]) + "_Dataset"
            dataset_class = getattr(dataset_module, formatted_name)
            class_weights = list(dataset_class.LABEL2WEIGHT.values())
        else:
            class_weights = None

        if task_config.task_type == "res":
            tasks_config[task] = dict(
                task_id=id,
                class_names=list(range(train_dataset.features['label'].length)),
                num_classes=train_dataset.features['label'].length,
                columns=task_config.columns,
                train_loader=train_loader,
                dev_loader=dev_loader,
                test_loader=test_loader,
                test_dataset=test_dataset,
                train_dataset=train_dataset,
                class_weights=class_weights,
                task_type=task_config.task_type
            )
        else:
            tasks_config[task] = dict(
                task_id=id,
                class_names=train_dataset.features['label'].names,
                num_classes=train_dataset.features['label'].num_classes,
                columns=task_config.columns,
                train_loader=train_loader,
                dev_loader=dev_loader,
                test_loader=test_loader,
                test_dataset=test_dataset,
                train_dataset=train_dataset,
                class_weights=class_weights,
                task_type=task_config.task_type
            )
    return tasks_config
