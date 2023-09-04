import datasets
from tqdm.auto import tqdm

PROMPTER = "<|prompter|>"
BOT = "<|assistant|>"
END = "<|endoftext|>"


def get_chat_dataset(T2T=False) -> datasets.Dataset:
    all_rows = []

    ds = datasets.load_dataset(
        "flozi00/conversations", split="train", cache_dir="./downloadcache"
    )

    if T2T is True:
        for x in ds:
            all_rows.append(x["conversations"])
        T2T_ROWS = []
        T2T_ANSWERS = []
        for row in tqdm(all_rows):
            messages = row.split(END)
            for m in range(0, len(messages) - 1):
                T2T_ROWS.append(END.join(messages[: m + 1]))
                T2T_ANSWERS.append(messages[m + 1])

        ds = datasets.Dataset.from_dict(
            {"conversations": T2T_ROWS, "answers": T2T_ANSWERS}
        )

    return ds
