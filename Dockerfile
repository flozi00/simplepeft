FROM nvcr.io/nvidia/pytorch:23.07-py3

RUN pip install --upgrade pip
RUN pip install transformers sentencepiece peft accelerate datasets optimum jupyterlab

CMD ["jupyter", "lab"]