from dataclasses import dataclass, field
from typing import Dict, List, Union

import torch

from ..languages import LANGUAGES
from transformers import AutoProcessor


@dataclass
class ASRDataCollator:
    processor: AutoProcessor
    wav_key: list = field(default_factory=list)
    locale_key: str = "locale"
    text_key: str = "sentence"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = []
        label_features = []

        for feature in features:
            myaudio = feature
            for k in self.wav_key:
                myaudio = myaudio[k]

            mytext = feature[self.text_key]
            mylang = feature[self.locale_key]

            extracted = self.processor.feature_extractor(
                myaudio,
                sampling_rate=16000,
                return_tensors="pt",
            )

            # check if feature extractor return input_features or input_values
            ft = (
                "input_values"
                if hasattr(extracted, "input_values")
                else "input_features"
            )

            # append to input_features
            input_features.append(
                {
                    ft: getattr(
                        extracted,
                        ft,
                    )[0].half()
                }
            )

            # set prefix tokens if possible
            try:
                values = list(LANGUAGES.values())
                prefix = mylang if mylang in values else LANGUAGES[mylang]
                self.processor.tokenizer.set_prefix_tokens(prefix)
            except Exception:
                pass

            # append to label_features
            label_features.append(
                {"input_ids": self.processor.tokenizer(mytext).input_ids}
            )

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding="longest",
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding="longest",
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch
