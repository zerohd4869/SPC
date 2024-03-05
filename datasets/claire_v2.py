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
CLAIRE (Roth,Anthonio,andSauer2022) dataset consists of manually clarified how-to guides from wikiHow6 with generated alternative clarifications and human plausibility judgements. The goal of plausible clarifications ranking task is to predict the continuous plausibility score on a scale from 1 (very implausible) to 5 (very plausible) given the clarification and its context. 
In our experiments, a special token pair (i.e., <e>and </e>) is introduced as the boundary of filler words.
"""

# Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# Add the licence for the dataset here if you can find it
_LICENSE = ""


class ClaireV2_Dataset(datasets.GeneratorBasedBuilder):
    """The CLAIRE dataset for Plausible Clarification Ranking task."""

    VERSION = datasets.Version("1.0.0")

    DEFAULT_CONFIG_NAME = "claire_v2"
    PATTERN_INFO = {
        "IMPLICIT REFERENCE": "IMPLICIT REFERENCE: In the original version of a sentence, there is an implicit reference to a previously mentioned entity. The revision makes this reference explicit.",
        "FUSED HEAD": "FUSED HEAD: In the original version, there is a noun phrase where the head noun is missing. The revision adds that noun.",
        "ADDED COMPOUND": "ADDED COMPOUND: The revision adds a compound modifier to a noun to make its meaning more specific.",
        "METONYMIC REFERENCE": "METONYMIC REFERENCE: In the original version, a noun is used in a metonymy. The revision makes the particular component or aspect of a noun explicit that is meant.", }

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
                    'label': datasets.features.Sequence(datasets.Value('float'), length=1, id=['score']),
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
                "label": [float(df["label"][id_])],
                "sentence1": str(df["sentence_e"][id_]),
                "sentence2": self.PATTERN_INFO[df["resolved_pattern"][id_]],
            }
