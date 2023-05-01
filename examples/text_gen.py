import datasets
from simplepeft.data.main import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks

BATCH_SIZE = 12
BASE_MODEL = "rwkv-430M-german-instructions"
PEFT_MODEL = "rwkv-430M-german-instructions"
TASK = Tasks.TEXT_GEN
LR = 1e-4

# generate an instruction dataset by using the instruction as prefix for the input and output
def add_prefix(example):
    example["text"] = (
        example["instruction"] + " " + example["input"] + " " + example["output"]
    )
    return example


def add_prefix_quad(example):
    example["text"] = (
        example["context"]
        + "\n" * 2
        + example["question"]
        + "\n" * 2
        + example["answers"]["text"][0]
    )
    return example


def add_prefix_gem(example):
    example["text"] = (
        "Fasse zusammen: \n"
        + example["source"]
        + "\n" * 2
        + example["target_aligned"]["de"]
    )

    return example


def get_dataset():
    ds = datasets.load_dataset(
        "philschmid/translated_tasks_de_google_52k", split="train"
    )
    ds = ds.map(add_prefix, remove_columns=ds.column_names)

    ds2 = datasets.load_dataset("deepset/germanquad", split="train")
    ds2 = ds2.map(add_prefix_quad, remove_columns=ds2.column_names)

    ds3 = datasets.load_dataset("gem", "wiki_lingua_german_de", split="train")
    ds3 = ds3.map(add_prefix_gem, remove_columns=ds3.column_names)

    ds4 = datasets.load_dataset("EleutherAI/lambada_openai", "de", split="test")

    ds: datasets.Dataset = datasets.concatenate_datasets([ds, ds2, ds3, ds4])

    # ds = ds.filter(lambda x: len(x["text"]) < 1024 * 3)
    ds = ds.shuffle(seed=42)

    return ds


def get_training_corpus(ds):
    dataset = ds
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["text"]


def main():
    # load model, processor and model_conf by using the get_model function
    model, processor, model_conf = get_model(
        task=TASK, model_name=BASE_MODEL, peft_name=PEFT_MODEL, use_peft=False
    )

    cv_data = get_dataset()

    # only run this if you want to train a new tokenizer for another language
    # processor = processor.train_new_from_iterator(get_training_corpus(cv_data), 128_000)
    # model.resize_token_embeddings(len(processor))

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,
        processor=processor,
        datas=cv_data,
        BATCH_SIZE=BATCH_SIZE,
        max_input_length=512,
        text_key="text",
    )

    # start training
    start_training(
        model=model,
        processor=processor,
        dloader=dloader,
        PEFT_MODEL=PEFT_MODEL,
        LR=LR,
        model_conf=model_conf,
        deepspeed=True,
    )


if __name__ == "__main__":
    main()
