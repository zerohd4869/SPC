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
EmojiEval (Barbieri et al. 2018) is designed for emoji prediction, which aims to predict its most likely emoji given a tweet. Its label set comprises 20 different emoji.
"""

# Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# Add the licence for the dataset here if you can find it
_LICENSE = ""


class TweetevalEmoji_Dataset(datasets.GeneratorBasedBuilder):
    """The EmojiEval dataset for Emoji Prediction task."""

    VERSION = datasets.Version("1.0.0")

    DEFAULT_CONFIG_NAME = "tweeteval_emoji"
    LABEL_NAMES = [
        "_red_heart_",
        "_smiling_face_with_hearteyes_",
        "_face_with_tears_of_joy_",
        "_fire_",
        "_sparkles_",
        "_two_hearts_",
        "_smiling_face_with_sunglasses_",
        "_camera_with_flash_",
        "_smiling_face_with_smiling_eyes_",
        "_camera_",
        "_Christmas_tree_",
        "_United_States_",
        "_blue_heart_",
        "_face_blowing_a_kiss_",
        "_winking_face_",
        "_beaming_face_with_smiling_eyes_",
        "_sun_",
        "_winking_face_with_tongue_",
        "_hundred_points_",
        "_purple_heart_"
    ]

    def _info(self):
        print("## DEFAULT_CONFIG_NAME：", self.DEFAULT_CONFIG_NAME)
        print("## _info")
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    # "scene_id": datasets.Value("string"),
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
                "id": str(id_),
                "label": str(df["label"][id_]),
                "sentence1": str(df["sentence1"][id_]),
                "sentence2": self.DEFAULT_CONFIG_NAME
            }
