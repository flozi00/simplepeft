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
        inputs = [
            f[self.source_key] for f in features
        ]  # list of strings (sentences) to be tokenized as inputs
        outputs = [
            f[self.target_key] for f in features
        ]  # list of strings (sentences) to be tokenized as labels

        # tokenize batched inputs
        model_inputs = self.tok(
            inputs,
            text_target=outputs,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # replace pad_token_id with -100 to ignore loss correctly
        model_inputs["labels"][model_inputs["labels"] == self.tok.pad_token_id] = -100

        return model_inputs


@dataclass
class CLMDataCollator:
    tok: AutoTokenizer
    text_key: str = "input"
    max_input_length: int = 1024

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        inputs = [
            f[self.text_key] for f in features
        ]  # list of strings (sentences) to be tokenized as inputs

        # tokenize batched inputs
        inputs = self.tok(
            inputs,
            max_length=self.max_input_length,
            padding="max_length" if self.tok.pad_token is not None else False,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # create batch dict with inputs and labels where labels = inputs
        batch = {
            "input_ids": inputs,
            "labels": inputs,
        }

        return batch
