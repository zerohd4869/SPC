# coding=utf-8

import os
import datasets
import pandas as pd

# Add BibTeX citation
_CITATION = """\
@InProceedings{}
"""

# Add description of the dataset here
_DESCRIPTION = """\
GoEmotions (Demszky et al. 2020) is a corpus of com- ments from Reddit, with human annotations to 27 emotion categories or neutral. It is used for fine-grained emotion detection. 
In this work, we remove all multi-label samples (nearly 16%) in the dataset to better evaluate the multi-class classification performance.
"""

# Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# Add the licence for the dataset here if you can find it
_LICENSE = ""


class GoEmotions_Dataset(datasets.GeneratorBasedBuilder):
    """The GoEmotions dataset for Fine-grained Emotion Detection task."""

    VERSION = datasets.Version("1.0.0")

    DEFAULT_CONFIG_NAME = "goemotions"
    LABEL_NAMES = ["neutral", "admiration", "gratitude", "approval", "amusement", "annoyance", "disapproval", "love", "curiosity", "anger", "optimism",
                   "confusion", "joy", "sadness", "surprise", "disappointment", "caring", "realization", "disgust", "excitement", "fear", "desire", "remorse",
                   "embarrassment", "nervousness", "relief", "pride", "grief"]

    def _info(self):
        print("## DEFAULT_CONFIG_NAME：", self.DEFAULT_CONFIG_NAME)
        print("## _info")
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sentence1": datasets.Value("string"),
                    "sentence2": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(num_classes=len(self.LABEL_NAMES), names=self.LABEL_NAMES),
                }
            ),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        print("## _split_generators")
        data_dir = os.path.join("./data/", self.DEFAULT_CONFIG_NAME)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.csv"),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.csv"),
                    "split": "dev"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.csv"),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, filepath, split):
        print("## _generate_examples")
        """Generate examples."""

        df = pd.read_csv(filepath)
        for id_ in range(len(df)):
            if id_ < 3: print("{}_example_{}: {}".format(split, id_, df.iloc[id_, :]))
            yield id_, {
                "id": str(df["id"][id_]),
                "label": str(df["label"][id_]),
                "sentence1": str(df["sentence1"][id_]),
                "sentence2": self.DEFAULT_CONFIG_NAME
            }
