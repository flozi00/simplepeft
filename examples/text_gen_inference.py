import gradio as gr
from threading import Thread
from transformers import AutoTokenizer, RwkvForCausalLM, TextIteratorStreamer

model = RwkvForCausalLM.from_pretrained("rwkv-430M-german-instructions")
tokenizer = AutoTokenizer.from_pretrained("rwkv-430M-german-instructions")


def predict(text):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    response = ""
    inputs = tokenizer(text, return_tensors="pt").input_ids
    generation_kwargs = dict(
        inputs=inputs,
        streamer=streamer,
        max_new_tokens=1024,
        use_cache=True,
        early_stopping=True,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        response += new_text
        yield response

    thread.join()
    return response


iface = gr.Interface(fn=predict, inputs="text", outputs="text")

iface.launch(enable_queue=True)
