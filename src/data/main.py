from src.data.ASRCollator import ASRDataCollator
from torch.utils.data import DataLoader
from src.data.TEXTCollator import CLMDataCollator, TextTextDataCollator

from src.utils import IS_WINDOWS, Tasks


def get_dataloader(
    task: str, processor: any, datas: any, BATCH_SIZE: int, **kwargs
) -> DataLoader:
    if task == Tasks.ASR:
        data_collator = ASRDataCollator(
            processor=processor,
            wav_key=kwargs.get("wav_key", ["audio", "array"]),
            locale_key=kwargs.get("locale_key", "locale"),
            text_key=kwargs.get("text_key", "sentence"),
        )
    elif task == Tasks.Text2Text:
        data_collator = TextTextDataCollator(
            tok=processor,
            source_key=kwargs.get("source_key", "source"),
            target_key=kwargs.get("target_key", "target"),
            max_input_length=kwargs.get("max_input_length", 1024),
            max_output_length=kwargs.get("max_output_length", 1024),
        )
    elif task == Tasks.TEXT_GEN:
        data_collator = CLMDataCollator(
            tok=processor,
            text_key=kwargs.get("text_key", "text"),
            max_input_length=kwargs.get("max_input_length", 1024),
        )

    dloader = DataLoader(
        datas,
        collate_fn=data_collator,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False if IS_WINDOWS else True,
        num_workers=0 if IS_WINDOWS else 2,
    )

    return dloader
