import datasets
from simplepeft.data.main import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks
import pandas as pd

BATCH_SIZE = 64
BASE_MODEL = "openai/whisper-small"
PEFT_MODEL = "whisper-small-german-lora-cv13"
TASK = Tasks.ASR
LR = 1e-5
CV_DATA_PATH = "../cv-corpus-13.0-2023-03-09/de/"

# generate the dataset from the common voice dataset saved locally and load it as a dataset object
# the dataset is filtered to only contain sentences with more than 5 characters and at least 2 upvotes and no downvotes
# the audio is casted to the Audio feature of the datasets library with a sampling rate of 16000
# the dataset is shuffled with a seed of 48
def get_dataset():
    df = pd.read_table(f"{CV_DATA_PATH}validated.tsv")
    df["audio"] = f"{CV_DATA_PATH}clips/" + df["path"].astype(str)
    df["down_votes"] = df["down_votes"].astype(int)
    df["up_votes"] = df["up_votes"].astype(int)
    df["sentence"] = df["sentence"].astype(str)

    mask = (
        (df["down_votes"] <= 0)
        & (df["up_votes"] >= 2)
        & (df["sentence"].str.len() >= 5)
    )
    df = df.loc[mask]

    d_sets = datasets.Dataset.from_pandas(df)

    d_sets = d_sets.cast_column("audio", datasets.features.Audio(sampling_rate=16000))
    d_sets = d_sets.shuffle(seed=48)

    print(d_sets)

    return d_sets


def main():
    cv_data = get_dataset()

    # get the model, processor and model_conf by configuration
    model, processor, model_conf = get_model(
        task=TASK,
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
    )

    # get the automatic dataloader for the given task, in this case the default arguments are working for data columns, otherwise they can be specified
    # check the **kwargs in the get_dataloader function in simplepeft/data/main.py for more information
    dloader = get_dataloader(
        task=TASK,
        processor=processor,
        datas=cv_data,
        BATCH_SIZE=BATCH_SIZE,
    )

    # start the training
    start_training(
        model=model,
        processor=processor,
        dloader=dloader,
        PEFT_MODEL=PEFT_MODEL,
        LR=LR,
        model_conf=model_conf,
    )


if __name__ == "__main__":
    main()
