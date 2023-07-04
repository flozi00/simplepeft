import datasets

PROMPTER = "<|prompter|>"
BOT = "<|assistant|>"
END = "<|endoftext|>"


def get_chat_dataset(T2T=False) -> datasets.Dataset:
    all_rows = []

    ds2 = datasets.load_dataset(
        "philschmid/translated_tasks_de_google_52k", split="train"
    )
    for row in ds2:
        if len(row["input"]) > 128:  # type: ignore
            all_rows.append(
                f'{PROMPTER}{row["instruction"]} {row["input"]} {END}{BOT}{row["output"]}{END}'  # type: ignore
            )

    ds2 = datasets.load_dataset(
        "argilla/databricks-dolly-15k-curated-multilingual", split="de+en"
    )
    for row in ds2:
        if len(row["context"]) > 128:  # type: ignore
            all_rows.append(
                f'{PROMPTER}{row["instruction"]} {row["context"]} {END}{BOT}{row["response"]}{END}'  # type: ignore
            )

    ds3 = datasets.load_dataset(
        "flozi00/openassistant-oasst1-flattened-filtered", split="train"
    ).filter(lambda example: example["lang"] in ["de", "en"])

    for x in ds3:
        all_rows.append(x["conversations"])

    ds4 = datasets.load_dataset(
        "SebastianBodza/Ger_WizardLM_evol_instruct_70k_V0", split="train"
    )
    for row in ds4:
        all_rows.append(
            f'{PROMPTER}{row["instructions"]}{END}{BOT}{row["outputs"]}{END}'  # type: ignore
        )

    ds5 = datasets.load_dataset("JosephusCheung/GuanacoDataset", split="train")
    for row in ds5:
        if "a" in {row["instruction"]} or "i" in {row["instruction"]}:
            all_rows.append(
                f'{PROMPTER}{row["input"]}\n{row["instruction"]}{END}{BOT}{row["output"]}{END}'  # type: ignore
            )

    if T2T is True:
        T2T_ROWS = []
        T2T_ANSWERS = []
        for row in all_rows:
            messages = row.split(END)
            for m in range(0, len(messages) - 1):
                T2T_ROWS.append(END.join(messages[: m + 1]))
                T2T_ANSWERS.append(messages[m + 1])

        ds = datasets.Dataset.from_dict(
            {"conversations": T2T_ROWS, "answers": T2T_ANSWERS}
        )
    else:
        ds = datasets.Dataset.from_dict({"conversations": all_rows})

    return ds
