from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from transformers import AutoTokenizer


@dataclass
class TextTextDataCollator:
    tok: AutoTokenizer
    source_key: str = "input"
    target_key: str = "output"
    max_input_length: int = 1024
    max_output_length: int = 1024

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        inputs = [f[self.source_key] for f in features]
        outputs = [f[self.target_key] for f in features]
        model_inputs = self.tok(
            inputs,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        labels = self.tok(
            outputs,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        labels[labels == self.tok.pad_token_id] = -100

        batch = {"input_ids": model_inputs, "labels": labels}
        return batch


@dataclass
class CLMDataCollator:
    tok: AutoTokenizer
    text_key: str = "input"
    max_input_length: int = 1024

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        inputs = [f[self.text_key] for f in features]

        inputs = self.tok(
            inputs,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        labels[labels == self.tok.pad_token_id] = -100

        inputs["labels"] = labels

        return inputs
