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
MELD(Poriaetal.2019) contain smulti-party conversation videos collected from Friends TV series, where two or more speakers are involved in a conversation. It is used to detect emotions in each utterance.
The MELD dataset contains many types of context, including dialogue, speaker, and multi-modal data. Different from other task-oriented methods, e.g., DialogueCRN (Hu, Wei, and Huai 2021), this work only considers the context-free textual utterance to better evaluate sentence classification performance.
"""

# Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# Add the licence for the dataset here if you can find it
_LICENSE = ""


class Meld_Dataset(datasets.GeneratorBasedBuilder):
    """The MELD dataset for Conversational Emotion Recognition task."""

    VERSION = datasets.Version("1.0.0")

    DEFAULT_CONFIG_NAME = "meld"
    LABEL_NAMES = [
        "neutral",
        "joy",
        "anger",
        "surprise",
        "sadness",
        "disgust",
        "fear"
    ]

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
                "id": str(df["Sr No."][id_]),
                "label": str(df["Emotion"][id_]),
                "sentence1": str(df["Utterance"][id_]),
                "sentence2": str(df["Speaker"][id_])
            }
