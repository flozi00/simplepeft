import datasets
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks
import pandas as pd

BATCH_SIZE = 1
BASE_MODEL = "RWKV/rwkv-raven-3b"
PEFT_MODEL = "rwkv-4-3b-german-assistant"
TASK = Tasks.TEXT_GEN
LR = 1e-4


def get_dataset():
    data_file = "OpenAssistant/oasst1"
    ds = datasets.load_dataset(data_file)
    df = pd.concat([ds["train"].to_pandas(), ds["validation"].to_pandas()], axis=0)  # type: ignore
    rows = {}
    message_ids = df["message_id"].values.tolist()
    message_tree_ids = df["message_tree_id"].values.tolist()
    parent_ids = df["parent_id"].values.tolist()
    texts = df["text"].values.tolist()
    roles = df["role"].values.tolist()
    langs = df["lang"].values.tolist()

    for i in range(df.shape[0]):
        # collect all trees
        message_id = message_ids[i]
        message_tree_id = message_tree_ids[i]
        parent_id = parent_ids[i]
        text = texts[i]
        role = roles[i]
        new_data = ("<|prompter|>" if role == "prompter" else "<|assistant|>") + text
        entry = dict(
            message_id=message_id, parent_id=parent_id, text=new_data, lang=langs[i]
        )
        if message_tree_id not in rows:
            rows[message_tree_id] = [entry]
        else:
            rows[message_tree_id].append(entry)

    all_rows = []

    for node_id in rows:
        # order responses in tree, based on message/parent relationship
        conversations = []

        list_msgs = rows[node_id]
        # find start
        while len(list_msgs):
            for i, leaf in enumerate(list_msgs):
                found = False
                parent_id = leaf["parent_id"]
                if parent_id is None:
                    # conversation starter
                    conversations.append(leaf)
                    found = True
                else:
                    for conv in conversations:
                        # find all conversations to add my message to
                        if (
                            parent_id in conv["message_id"]
                            and parent_id != conv["message_id"][-len(parent_id) :]
                        ):
                            # my message doesn't follow conversation
                            continue
                        if parent_id == conv["message_id"][-len(parent_id) :]:
                            # my message follows conversation, but fork first, so another follow-on message can do same
                            conversations.append(conv.copy())
                            conv[
                                "text"
                            ] += f"""
    {leaf['text']}
    """
                            conv["message_id"] += leaf["message_id"]
                            found = True
                            break
                if found:
                    # my content was used, so nuke from list
                    del list_msgs[i]
                    break

        # now reduce down to final conversations, find the longest chains of message ids
        for i, conv in enumerate(conversations):
            for j, conv2 in enumerate(conversations):
                if i == j:
                    continue
                if conv["message_id"] and conv2["message_id"]:
                    assert conv["message_id"] != conv2["message_id"]
                    # delete the shorter conversation, if one contains the other
                    if conv["message_id"] in conv2["message_id"]:
                        conv["message_id"] = None
                    if conv2["message_id"] in conv["message_id"]:
                        conv2["message_id"] = None
        conversations = [c for c in conversations if c["message_id"]]
        for c in conversations:
            if c["lang"] == "de":
                all_rows.append(c["text"])

    ds2 = datasets.load_dataset(
        "philschmid/translated_tasks_de_google_52k", split="train"
    )
    for row in ds2:
        all_rows.append(
            f"<|prompter|>{row['instruction']} {row['input']}<|assistant|>{row['output']}"  # type: ignore
        )

    ds2 = datasets.load_dataset(
        "argilla/databricks-dolly-15k-curated-multilingual", split="de"
    )
    for row in ds2:
        all_rows.append(
            f"<|prompter|>{row['context']}\n\n{row['instruction']}<|assistant|>{row['response']}"  # type: ignore
        )

    print(len(all_rows))

    ds = datasets.Dataset.from_dict({"text": all_rows})

    return ds


def main():
    # load model, processor and model_conf by using the get_model function
    model, processor, model_conf = get_model(
        task=TASK, model_name=BASE_MODEL, peft_name=PEFT_MODEL, use_peft=False  # type: ignore
    )

    cv_data = get_dataset()

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=processor,
        datas=cv_data,
        BATCH_SIZE=BATCH_SIZE,
        max_input_length=2048,
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
    )


if __name__ == "__main__":
    main()
