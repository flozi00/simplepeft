import datasets
import torch
from simplepeft.data import get_dataloader
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks
import pandas as pd
from jiwer import wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from unidecode import unidecode

BATCH_SIZE = 8
BASE_MODEL = "facebook/wav2vec2-large-xlsr-53"
BASE_PROCESSOR = "aware-ai/wav2vec2-xls-r-300m-german"
PEFT_MODEL = "wav2vec2-large-xlsr-german-cv14"
TASK = Tasks.ASR
LR = 3e-4


def normalize_text(batch):
    text = batch["sentence"]
    couples = [
        ("ä", "ae"),
        ("ö", "oe"),
        ("ü", "ue"),
        ("Ä", "Ae"),
        ("Ö", "Oe"),
        ("Ü", "Ue"),
    ]

    # Replace special characters with their ascii equivalent
    for couple in couples:
        text = text.replace(couple[0], f"__{couple[1]}__")
    text = text.replace("ß", "ss")
    text = unidecode(text)

    # Replace the ascii equivalent with the original character after unidecode
    for couple in couples:
        text = text.replace(f"__{couple[1]}__", couple[0])

    batch["sentence"] = text
    return batch


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
    cv_data = get_dataset().map(normalize_text)
    all_chars = set(" ".join(cv_data["sentence"]))

    processor = Wav2Vec2Processor.from_pretrained(BASE_PROCESSOR)
    vocab_size = len(processor.tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        vocab_size=vocab_size,
        # target_lang="deu",
        # ignore_mismatched_sizes=True,
    ).cuda()

    model.config.ctc_loss_reduction = "mean"

    model.config.update(
        {
            "feat_proj_dropout": 0,
            "attention_dropout": 0,
            "hidden_dropout": 0,
            "final_dropout": 0,
            "mask_time_prob": 0,
            "mask_time_length": 0,
            "mask_feature_prob": 0,
            "mask_feature_length": 0,
            "layerdrop": 0,
            "ctc_loss_reduction": "mean",
            "pad_token_id": processor.tokenizer.pad_token_id,
            "activation_dropout": 0,
        }
    )

    def eval_func_w2v():
        model.eval()
        trans = []
        vals = []
        for xyz in range(0, BATCH_SIZE * 16, BATCH_SIZE):
            audios = [
                cv_data[index]["audio"]["array"]
                for index in range(xyz, xyz + BATCH_SIZE)
            ]
            for index in range(xyz, xyz + BATCH_SIZE):
                vals.append(cv_data[index]["sentence"])
            inputs = (
                processor(
                    audios, sampling_rate=16000, return_tensors="pt", padding=True
                )
                .to(model.device)
                .input_values
            )
            with torch.no_grad():
                logits = model(input_values=inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            trans.extend(transcription)

        model.train()

        return wer(vals, trans) * 100

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
