# t2t-tuner

Convenient Text-to-Text Training for Transformers

```shell
pip install t2t-tuner
```

## Features

* Easy training for text-to-text generation tasks
* Training methods/features:
  * Supervised fine-tuning 
  * Gradient checkpointing
  * Model parallelism
  * Soft prompt tuning ([based on this paper](https://arxiv.org/abs/2104.08691))
  * Freeze encoder/decoder/embeddings
  * Print model summary
* Based on the wonderful [HuggingFace Transformers](https://github.com/huggingface/transformers) library. Tested models: T5-based models. In theory, it should work with other models that support [AutoModelForSeq2SeqLM](https://huggingface.co/transformers/model_doc/auto.html#automodelforseq2seqlm) as well

This work is based on HuggingFace's [run_translation.py](https://github.com/huggingface/transformers/tree/master/examples/pytorch/translation) script for text-to-text generation tasks. It provides (what I feel is) a more convenient interface to training and inferencing text-to-text generation models, along with better access to some features and new features that I added in myself.

## Examples

Simple snippet:

```python
import t2t

trainer_arguments = t2t.TrainerArguments(model_name_or_path="t5-small",
                                         train_file=YOUR_DATASET)

trainer = t2t.Trainer(arguments=trainer_arguments)

trainer.train()
```

For more concrete examples, check out the notebooks linked below:

* [Simple example](examples/tldr.ipynb)
* [Soft Prompt Tuning](examples/soft_prompt_tuning.ipynb)

