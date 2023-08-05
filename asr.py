import datasets
import torch
from simplepeft.data import get_dataloader
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks
import pandas as pd
from jiwer import wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

BATCH_SIZE = 16
BASE_MODEL = "aware-ai/wav2vec2-xls-r-300m-german"
PEFT_MODEL = "wav2vec2-300m-german-cv14"
TASK = Tasks.ASR
LR = 1e-5


# generate the dataset from the common voice dataset saved locally and load it as a dataset object
# the dataset is filtered to only contain sentences with more than 5 characters and at least 2 upvotes and no downvotes
# the audio is casted to the Audio feature of the datasets library with a sampling rate of 16000
def get_dataset() -> datasets.Dataset:
    CV_DATA_PATH = "./cv-corpus-14.0-2023-06-23/de/"
    df = pd.read_table(filepath_or_buffer=f"{CV_DATA_PATH}validated.tsv")
    df["audio"] = f"{CV_DATA_PATH}clips/" + df["path"].astype(dtype=str)
    df["down_votes"] = df["down_votes"].astype(dtype=int)
    df["up_votes"] = df["up_votes"].astype(dtype=int)
    df["sentence"] = df["sentence"].astype(dtype=str)

    mask = (
        (df["down_votes"] <= 0)
        & (df["up_votes"] >= 2)
        & (df["sentence"].str.len() >= 5)
    )
    df = df.loc[mask]

    d_sets = datasets.Dataset.from_pandas(df=df)

    d_sets = d_sets.cast_column(
        column="audio", feature=datasets.features.Audio(sampling_rate=16000)
    )

    return d_sets


def main():
    cv_data = get_dataset()

    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    vocab_size = len(processor.tokenizer)
    model = (
        Wav2Vec2ForCTC.from_pretrained(BASE_MODEL, vocab_size=vocab_size).cuda().half()
    )

    def eval_func_whisper():
        model.eval()
        trans = []
        vals = []
        for xyz in range(BATCH_SIZE):
            inputs = processor(cv_data[xyz]["audio"]["array"], return_tensors="pt")
            input_features = inputs.input_features.to(model.device).half()
            generated_ids = model.generate(inputs=input_features)
            transcription = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            trans.append(transcription)
            vals.append(cv_data[xyz]["sentence"])

        print(wer(vals, trans))
        model.train()

    def eval_func_w2v():
        model.eval()
        trans = []
        vals = []
        for xyz in range(256):
            inputs = (
                processor(
                    cv_data[xyz]["audio"]["array"],
                    sampling_rate=16000,
                    return_tensors="pt",
                )
                .to(model.device)
                .input_values.half()
            )
            with torch.no_grad():
                logits = model(input_values=inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            trans.append(transcription)
            vals.append(cv_data[xyz]["sentence"])

        print(wer(vals, trans) * 100)
        model.train()

    # get the automatic dataloader for the given task, in this case the default arguments are working for data columns, otherwise they can be specified
    # check the **kwargs in the get_dataloader function in simplepeft/data/main.py for more information
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=processor,
        datas=cv_data,
        max_audio_in_seconds=16,
        BATCH_SIZE=BATCH_SIZE,
        batch_size=BATCH_SIZE,
    )

    # start the training
    start_training(
        model=model,
        processor=processor,
        dloader=dloader,
        PEFT_MODEL=PEFT_MODEL,
        LR=LR,
        callback=eval_func_w2v,
    )


if __name__ == "__main__":
    main()
