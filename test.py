from transformers import AutoProcessor, Wav2Vec2ForCTC
import datasets
from bitsandbytes.optim import PagedAdamW32bit
import librosa

BASE_MODEL = "aware-ai/wav2vec2-xls-r-300m-german"
LR = 1e-5

ds = datasets.load_dataset("mozilla-foundation/common_voice_13_0", "de", split="train")

try:
    print(ds[0])
except Exception as e:
    print(e)

print("load audio")
audio, sr = librosa.load(ds[0]["audio"]["path"], sr=16000)


model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL)
processor = AutoProcessor.from_pretrained(BASE_MODEL)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
inputs["labels"] = processor(text=ds[0]["text"], return_tensors="pt").input_ids


optim = PagedAdamW32bit(model.parameters(), lr=LR)

for xyz in range(100):
    optim.zero_grad()
    output = model(return_dict=True, **inputs)
    loss = output.loss
    loss.backward()
    optim.step()
    print(loss)
