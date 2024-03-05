# coding=utf-8

import json
import os
import datasets

# Add BibTeX citation
_CITATION = """\
@InProceedings{}
"""

# Add description of the dataset here
_DESCRIPTION = """\
STS-B (Cer et al. 2017) is a collection of English sentence pairs drawn from news headlines, video and image captions, and natural language inference data. The semantic similarity prediction task is to predict the semantic textual similarity score from 0 (very dissimilar) to 5 (very similar) given each sentence pair.
"""

# Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

#  Add the licence for the dataset here if you can find it
_LICENSE = ""


class GlueStsb_Dataset(datasets.GeneratorBasedBuilder):
    """The STS-B dataset for Semantic Similarity Prediction task."""

    VERSION = datasets.Version("1.0.0")

    DEFAULT_CONFIG_NAME = "glue_sts_b"

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
                    "label": datasets.features.Sequence(datasets.Value('float'), length=1, id=['class']),
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
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "validation.jsonl"),
                    "split": "dev"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)

                yield key, {
                    "id": str(key),
                    "sentence1": data["sentence1"],
                    "sentence2": data["sentence2"],
                    "label": [float(data["score"])],
                }
