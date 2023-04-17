import datasets
from simplepeft.data.main import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks
import pandas as pd

BATCH_SIZE = 64
BASE_MODEL = "openai/whisper-large-v2"
PEFT_MODEL = "whisper-large-v2-german-lora-cv13-simplepeft"
TASK = Tasks.ASR
LR = 1e-5
CV_DATA_PATH = "./cv-corpus-13.0-2023-03-09/de/"


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
    model, processor, model_conf = get_model(
        task=TASK,
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
    )

    cv_data = get_dataset()

    dloader = get_dataloader(
        task=TASK,
        processor=processor,
        datas=cv_data,
        BATCH_SIZE=BATCH_SIZE,
    )

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
