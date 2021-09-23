# t2t-tuner

Convenient Text-to-Text Training for Transformers

```shell
pip install t2t-tuner
```

Requires PyTorch: either follow [PyTorch installation instructions](https://pytorch.org/get-started/locally/) or [use a PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch).

## Features

* Easy training for text-to-text generation tasks
* Training methods/features:
  * Supervised fine-tuning 
  * Gradient checkpointing
  * Model parallelism
  * Soft prompt tuning ([based on this paper](https://arxiv.org/abs/2104.08691))
  * Freeze encoder/decoder/embeddings
  * Print model summary
* Based on the wonderful [HuggingFace Transformers](https://github.com/huggingface/transformers) library. Tested on T5-based models. In theory, it should work with other models that support [AutoModelForSeq2SeqLM](https://huggingface.co/transformers/model_doc/auto.html#automodelforseq2seqlm) as well

This work is based on HuggingFace's [run_translation.py](https://github.com/huggingface/transformers/tree/master/examples/pytorch/translation) script for text-to-text generation tasks. I decided I want a more more convenient interface for training and inferencing, along with access to things like gradient checkpointing and model parallel to fit larger models - these are already in the HuggingFace library but not exposed in the script. I also added in some features that I wanted (prompt tuning, model summary) and wrapped it as a library that can be pip installed. 

## Examples

Simple snippet:

```python
import t2t

trainer_arguments = t2t.TrainerArguments(model_name_or_path="t5-small",
                                         train_file=YOUR_DATASET)

trainer = t2t.Trainer(arguments=trainer_arguments)

# train without validation
trainer.train(valid=False)
```

For more concrete examples, check out the notebooks linked below:

* [Simple example](examples/tldr.ipynb)
* [Simple example on Colab](https://colab.research.google.com/drive/1_BsldxfPl6lVh2dB9VLOvARRxfswfIzL?usp=sharing)
* [Soft Prompt Tuning](examples/soft_prompt_tuning.ipynb)
* [Gradient checkpointing](examples/gradient_checkpointing.ipynb)
* [Model parallelism](examples/model_parallel.ipynb)

Data format:

```json
{"translation": {"s": "TEXT", "t": "LABEL"}}
```

* The format of data is json-lines, following HuggingFace original script. Each example is one line
* Define the source and target IDs in `TrainingArguments.source_id` and `TrainingArguments.target_id` (defaults to `s` and `t`)
* Include the prefix in the data file, or define the prefix to prepend to the text in `TrainingArguments.prefix`
* [Example notebook for data preprocessing from CSV file](sample_data/make_seq2seq_dataset.ipynb)

## Training Large Models

Using this library, you can fine-tune the [T5 11b checkpoints](https://huggingface.co/models?search=11b) quite easily with the following settings:

* Batch size 1 + gradient accumulation to make up to whatever batch size you need
* Batch size of 8 is possible with gradient checkpointing, but doesn't improve the speed
* About 128GB of VRAM: 8x 16GB or 4x 32GB GPU (such as V100)
* FP32 (no need for mixed precision)
  * FP16 would actually be better, but the pretrained T5 checkpoints don't play well with FP16
  * Likely reason: the existing activations are too large ([github issue tracking](https://github.com/huggingface/transformers/pull/10956#issuecomment-813162960), [some more info](https://discuss.huggingface.co/t/mixed-precision-for-bfloat16-pretrained-models/5315))

![Model parallel T5-11b](images/model_parallel.jpg)

Note that depending on your system, the loading time for the checkpoint (46GB) can be quite long.

## ONNX RT

```shell
pip install torch-ort -f https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch190.cu111.html

python -m torch_ort.configure
```

## Development

**Building Package**

```shell
python3 -m pip install --upgrade build twine
python3 -m build
python3 -m twine upload dist/*
```

## Disclaimers

This library as developed as a personal project for my own use. Please feel free to fork or use it for your own purposes as well. I will not take responsibility for any mishaps that occur as a result of this library's usage. 
