# SPC

This repository contains the official code for the paper [Structured Probabilistic Coding](https://arxiv.org/abs/2312.13933), accepted at [AAAI 2024 Oral].

## Highlights

Structured Probabilistic Coding (SPC) is a supervised representation learning technology serving as an encoder-only probabilistic coding framework with structured regularization from the target space.

By learning compact and informative representations from input related to the target task, SPC enhances the generalization ability of pre-trained language models for better language understanding.

Experimental results on 12 natural language understanding tasks demonstrate that SPC effectively improves the performance of PLMs for classification and regression.

## News
- [TODO]: Release the model chechpoints of SPC.
- **[Mar 2024]**: Added support for the multi-task version of SPC.
- **[Feb 2024]**: Code is available on [GitHub](https://github.com/zerohd4869/SPC).
- **[Dec 2023]**: Paper is available on [arXiv](https://arxiv.org/abs/2312.13933).
- **[Dec 2023]**: Paper is accepted by AAAI 2024 (Oral).

## Quick Start

1. Clone the repository
```
git clone https://github.com/zerohd4869/SPC.git
cd /SPC
```

2. Download the data and pre-trained model parameters
Download 12 datasets mentioned in the paper from [here](https://drive.google.com/file/d/161eu3T7XS-DUl57pQUgtvYG6eGBwLM5H/view?usp=sharing), and extract the files into the `/SPC/data/` directory.
This repo already contains 7 of these datasets by default, so this step is optional.

Download the `roberta-base` model parameters from [here](https://huggingface.co/FacebookAI/roberta-base) and place them in the `/SPC/ptms/roberta-base/` directory.
SPC is a backbone-free representation learning method. When using it, you could choose an appropriate backbone model and initialized parameter checkpoints for your task or dataset.

2. Install dependencies
``` 
# env: Python 3.7.16, Tesla A100 80GB
pip install -r spc_requirements.txt
```

3. Run examples

For classification:
```
# EmojiEval dataset
nohup bash script/run_train_emojieval.sh >  spc_roberta_emojieval.out &

# EmotionEval dataset
nohup bash script/run_train_emotioneval.sh >  spc_roberta_emotioneval.out &

# HatEval dataset
nohup bash script/run_train_hateval.sh >  spc_roberta_hateval.out &

# IronyEval dataset
nohup bash script/run_train_ironyeval.sh >  spc_roberta_ironyeval.out &

# OffensEval dataset
nohup bash script/run_train_offenseval.sh >  spc_roberta_offenseval.out &

# SentiEval dataset
nohup bash script/run_train_sentieval.sh >  spc_roberta_sentieval.out &

# StanceEval dataset
nohup bash script/run_train_stanceeval.sh >  spc_roberta_stanceeval.out &

# ISEAR dataset
nohup bash script/run_train_isear.sh >  spc_roberta_isear.out &

# MELD dataset
nohup bash script/run_train_meld.sh >  spc_roberta_meld.out &

# GoEmotions dataset
nohup bash script/run_train_goemotions.sh >  spc_roberta_goemotions.out &
```

For regression:
```
# STS-B dataset
nohup bash script/run_train_sbsb.sh >  spc_roberta_stsb.out &

# CLAIRE dataset
nohup bash script/run_train_claire.sh >  spc_roberta_claire.out &
```

## Additional Recipes

**Apply for a new task/dataset**
1. Data preparation and loading script. Download the new dataset (take `NewDataset` as an example) and place the unzip files in the `/SPC/data/` directory. Add the label information of this dataset to the dictionary file `SPC/data/task2label.json`.
Then, refer to the template `/SPC/datasets/new_dataset_script.py` to write the corresponding reading script for the dataset and place the file in the `/SPC/datasets/` directory. Also, add the dataset and task information to the file `SPC/task.py` at the corresponding location.

2. Refer to the Quick Start section above to write the corresponding sh script and run it.

**Apply all tasks in a multi-task paradigm**

```
# 6 tasks/datasets in TweetEval
nohup bash script/run_train_mtl_tweeteval.sh >  spc_roberta_mtl_tweeteval.out &
```

## Citation

If you are interested in this work and want to use the code in this repo, please **star** this repo and **cite** it as:

```
@inproceedings{hu2024structured,
  title={Structured Probabilistic Coding},
  author={Dou Hu and Lingwei Wei and Yaxin Liu and Wei Zhou and Songlin Hu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```