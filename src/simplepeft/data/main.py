from .AUDIOCollator import ASRDataCollator, TTSDataCollator
from torch.utils.data import DataLoader
from ..data.TEXTCollator import CLMDataCollator, TextTextDataCollator

from ..utils import IS_WINDOWS, Tasks


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
    elif task == Tasks.TTS:
        data_collator = TTSDataCollator(
            processor=processor,
            reduction_factor=kwargs.get("reduction_factor", 2),
            wav_key=kwargs.get("wav_key", ["audio", "array"]),
            text_key=kwargs.get("text_key", "sentence"),
            speaker_model=kwargs.get("speaker_model", None),
        )

    dloader = DataLoader(
        datas,
        collate_fn=data_collator,
        batch_size=BATCH_SIZE,
        pin_memory=False,
        num_workers=0,
    )

    return dloader
