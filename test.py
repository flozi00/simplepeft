import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")

inputs = tokenizer("This is an example.", return_tensors="pt").input_ids
# Feed everything to the model

tokens = len(inputs[0])
print(tokens)

size = 2
steps = int(tokens / size)
stepper = 0

chunks = []

for i in range(steps+1):
    chunks.append(inputs[:, stepper : stepper + size])
    stepper += size

print(chunks)

print(inputs)