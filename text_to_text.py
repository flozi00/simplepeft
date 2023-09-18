from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.utils import Tasks
import simplepeft.train.train
import datasets

simplepeft.train.train.ACCUMULATION_STEPS = 16


BATCH_SIZE = 16
BASE_MODEL = "t5-small"
PEFT_MODEL = "t5-small-llm-tasks"
TASK = Tasks.Text2Text
LR = 1e-4


def main():
    ds = datasets.load_dataset("flozi00/LLM-Task-Classification", split="train")

    model, processor = get_model(
        task=TASK,  # type: ignore
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
        use_peft=True,
        use_bnb=False,
    )

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=processor,
        datas=ds,
        BATCH_SIZE=BATCH_SIZE,
        source_key="text",
        target_key="label",
        max_input_length=512,
        max_output_length=5,
    )

    # start training
    simplepeft.train.train.start_training(
        model=model,
        processor=processor,
        dloader=dloader,
        PEFT_MODEL=PEFT_MODEL,
        LR=LR,
    )


if __name__ == "__main__":
    main()
