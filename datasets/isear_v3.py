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
ISEAR (Scherer and Wallbott 1994) is from International Survey On Emotion Antecedents And Reactions project and contains reports on seven emotions each by close to 3000 respondents in 37 countries on all 5 continents. It aims to predict the emotion reaction. 
Due to the lack of a predefined split in the original dataset, we randomly split the dataset into train/valid/test set in a ratio of 4:1:5 based on the label distribution.
"""

# Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# Add the licence for the dataset here if you can find it
_LICENSE = ""


class IsearV3_Dataset(datasets.GeneratorBasedBuilder):
    """The ISEAR dataset for Emotion Reaction Prediction task."""

    VERSION = datasets.Version("1.0.0")

    DEFAULT_CONFIG_NAME = "isear_v3"
    LABEL_NAMES = [
        "shame",
        "disgust",
        "sadness",
        "anger",
        "fear",
        "joy",
        "guilt"
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
                "id": str(df["id"][id_]),
                "label": str(df["label"][id_]),
                "sentence1": str(df["sentence1"][id_]),
                "sentence2": self.DEFAULT_CONFIG_NAME
            }
