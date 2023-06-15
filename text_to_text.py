import datasets
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks

BATCH_SIZE = 2
BASE_MODEL = "t5-small"
PEFT_MODEL = "t5-small-relevance"
TASK = Tasks.Text2Text
LR = 1e-5


def get_dataset():
    dprdata = datasets.load_dataset('deepset/germandpr',split="train" ,use_auth_token=True)
    dataset = {"prompt": [], "target": []}
    # training set:
    for data in dprdata:
        query = data['question']
        positive_passages = data['positive_ctxs']["text"]
        negative_passages = data['hard_negative_ctxs']["text"]
    
        for entry in positive_passages:
            input_text = "Query: "  + query + " Context: " + entry
            label_text = "relevant"

            dataset["prompt"].append(input_text)
            dataset["target"].append(label_text)
            
        
        for entry in negative_passages:
            input_text = "Query: "  + query + " Context: " + entry
            label_text = "irrelevant"

            dataset["prompt"].append(input_text)
            dataset["target"].append(label_text)   

    return dataset

def main():
    # load model, processor and model_conf by using the get_model function
    model, processor, model_conf = get_model(
        task=TASK, # type: ignore
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
    )

    ds = datasets.Dataset.from_dict(get_dataset())
    print(ds)

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK, # type: ignore
        processor=processor,
        datas=ds,
        BATCH_SIZE=BATCH_SIZE,
        source_key="prompt",
        target_key="target",
        max_input_length=512,
        max_output_length=8,
    )

    # start training
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
