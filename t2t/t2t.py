import os
from dataclasses import dataclass, field
from typing import Optional
import torch
import numpy as np
from datasets import load_dataset, load_metric
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


@dataclass
class TrainerArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(default="./saved_model/")
    metric: Optional[str] = field(default="sacrebleu")
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default="~/cache/")
    model_name_or_path: str = field(default="t5-small")
    pad_to_max_length: Optional[bool] = field(default=False)
    max_source_length: Optional[int] = field(default=128)
    max_target_length: Optional[int] = field(default=128)
    ignore_pad_token_for_loss: Optional[bool] = field(default=True)
    preprocessing_num_workers: Optional[int] = field(default=1)
    overwrite_cache: Optional[bool] = field(default=False)
    fp16: Optional[bool] = field(default=False)
    pad_to_multiple_of: Optional[int] = field(default=8)
    predict_with_generate: Optional[bool] = field(default=True)
    resume_from_checkpoint: Optional[str] = field(default=None)
    generation_num_beams: Optional[int] = field(default=4)
    source_id: Optional[str] = field(default="s")
    target_id: Optional[str] = field(default="t")
    prefix: Optional[str] = field(default="")
    gradient_checkpointing: Optional[bool] = field(default=False)
    dropout_rate: Optional[float] = field(default=0.1)
    model_parallel_gpus: Optional[int] = field(default=1)
    logging_strategy: Optional[str] = field(default="epoch")
    save_strategy: Optional[str] = field(default="epoch")
    save_total_limit: Optional[int] = field(default=2)


class Trainer:
    def __init__(self, arguments):
        self.arguments = arguments
        self.last_checkpoint = None
        if (
            os.path.isdir(self.arguments.output_dir)
            and not self.arguments.overwrite_output_dir
        ):
            self.last_checkpoint = get_last_checkpoint(self.arguments.output_dir)
            if (
                self.last_checkpoint is None
                and len(os.listdir(self.arguments.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({self.arguments.output_dir}) already exists and is not empty. "
                )
        set_seed(self.arguments.seed)
        self.metric = load_metric(self.arguments.metric)
        self.load_dataset()
        self.load_model()
        self.is_prompt_tuning_only = False

    def load_dataset(self):
        if self.arguments.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            self.raw_datasets = load_dataset(
                self.arguments.dataset_name,
                self.arguments.dataset_config_name,
                cache_dir=self.arguments.cache_dir,
            )
        else:
            data_files = {}
            if self.arguments.train_file is not None:
                data_files["train"] = self.arguments.train_file
                extension = self.arguments.train_file.split(".")[-1]
            if self.arguments.validation_file is not None:
                data_files["valid"] = self.arguments.validation_file
                extension = self.arguments.validation_file.split(".")[-1]
            if self.arguments.test_file is not None:
                data_files["test"] = self.arguments.test_file
                extension = self.arguments.test_file.split(".")[-1]
            self.raw_datasets = load_dataset(
                extension, data_files=data_files, cache_dir=self.arguments.cache_dir
            )

    def freeze_params(self, module):
        for par in module.parameters():
            par.requires_grad = False

    def unfreeze_params(self, module):
        for par in module.parameters():
            par.requires_grad = True

    def freeze(self, encoder=False, decoder=False, embeddings=False):
        if encoder:
            self.freeze_params(self.model.encoder)
        else:
            self.unfreeze_params(self.model.encoder)
        if decoder:
            self.freeze_params(self.model.decoder)
        else:
            self.unfreeze_params(self.model.decoder)
        if embeddings:
            self.freeze_params(self.model.shared)
        else:
            self.unfreeze_params(self.model.shared)

    def round_up(self, x, to=8):
        x, to = int(x), int(to)
        return int((x + to - 1) & (-1 * to))

    def resize_token_embeddings_layer(self, round_up=True):
        vocab_size = len(self.tokenizer)
        if round_up:
            vocab_size = self.round_up(vocab_size, to=8)
        self.model.resize_token_embeddings(vocab_size)

    def add_new_tokens(self, additional_special_tokens):
        """List of additional special tokens to add"""
        special_tokens_dict = {"additional_special_tokens": additional_special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.resize_token_embeddings_layer()

    def freeze_embedding_params_partial(
        self, layer, weight_indices, weight_hook_handle=None
    ):
        # https://github.com/galidor/PyTorchPartialLayerFreezing/blob/main/partial_freezing.py
        if weight_hook_handle is not None:
            weight_hook_handle.remove()
        if weight_indices == [] or weight_indices is None:
            return
        if not isinstance(layer, torch.nn.Embedding):
            raise ValueError("layer must be a valid Embedding layer")
        if max(weight_indices) >= layer.weight.shape[0]:
            raise IndexError(
                "weight_indices must be less than the number output channels"
            )

        def freezing_hook_weight_full(grad, weight_multiplier):
            return grad * weight_multiplier.to(grad.device)

        weight_multiplier = torch.ones(layer.weight.shape[0])
        weight_multiplier[weight_indices] = 0
        weight_multiplier = weight_multiplier.view(-1, 1)

        def freezing_hook_weight(grad):
            return freezing_hook_weight_full(grad, weight_multiplier)

        weight_hook_handle = layer.weight.register_hook(freezing_hook_weight)
        return weight_hook_handle

    def reinit_new_embeddings(self, layer, weight_indices):
        new_embed_range_start = weight_indices[-1] + 1
        for param in layer.parameters():
            num_embeds = param.data.shape[0]
            mean_embed = torch.mean(param.data, axis=0)
            for i in range(new_embed_range_start, num_embeds):
                param.data[i] = mean_embed

    def convert_to_prompt_tuning(self, prompt_list, device="cuda:0", use_hook=True):
        if use_hook:
            # Freeze whole model
            for par in self.model.parameters():
                par.requires_grad = False
            # Unfreeze embedding layer
            for par in self.model.shared.parameters():
                par.requires_grad = True
            # Add new tokens and resize embedding layer
            old_vocab_size = len(self.tokenizer)
            self.add_new_tokens(prompt_list)
            freeze_range = list(range(0, old_vocab_size))
            # Freeze (zero out gradient) for all but the newly added tokens
            self.freeze_embedding_params_partial(
                self.model.shared, weight_indices=freeze_range
            )
            self.reinit_new_embeddings(self.model.shared, weight_indices=freeze_range)
            self.is_prompt_tuning_only = True
        else:
            raise NotImplementedError("Non-hook method not implemented")

    def count_parameters(self, trainable_only=False, return_M=False):
        if trainable_only:
            params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            params = sum(p.numel() for p in self.model.parameters())
        if return_M:
            params = round(params / 1e6, 1)
        return params

    def model_summary(self, no_print=False):
        _string_list = [
            "Summary",
            "=======",
            "- model name: " + self.arguments.model_name_or_path,
            "- model params:",
            "  - train: "
            + str(self.count_parameters(trainable_only=True, return_M=True))
            + " M",
            "  - total: "
            + str(self.count_parameters(trainable_only=False, return_M=True))
            + " M",
            "  - vocab: " + str(len(self.tokenizer)),
            "- prompt tuning only: " + str(self.is_prompt_tuning_only),
        ]
        _string = "\n".join(_string_list)
        if no_print:
            return _string
        else:
            print(_string)

    def load_model(self):
        print(
            "Loading",
            self.arguments.model_name_or_path,
            "(for large models, this might take a while)",
        )
        print("Files will be cached at:", self.arguments.cache_dir)
        print(
            "Ensure this directory is persistent if you do not want to download model files again!"
        )
        self.config = AutoConfig.from_pretrained(
            self.arguments.model_name_or_path,
            cache_dir=self.arguments.cache_dir,
            gradient_checkpointing=self.arguments.gradient_checkpointing,
            dropout_rate=self.arguments.dropout_rate,
            use_cache=not self.arguments.gradient_checkpointing,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.arguments.model_name_or_path,
            cache_dir=self.arguments.cache_dir,
            use_fast=True,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.arguments.model_name_or_path,
            from_tf=bool(".ckpt" in self.arguments.model_name_or_path),
            config=self.config,
            cache_dir=self.arguments.cache_dir,
        )
        if self.arguments.model_parallel_gpus > 1:
            device_map = {}
            total_num_layers = self.config.num_layers
            layer_per_gpu = int(total_num_layers / self.arguments.model_parallel_gpus)
            for i in range(self.arguments.model_parallel_gpus):
                layers = list(range(i * layer_per_gpu, (i + 1) * layer_per_gpu))
                device_map[i] = layers
                print("Place layers", layers, "on device", i)
            self.model.parallelize(device_map)
        self.padding = "max_length" if self.arguments.pad_to_max_length else False

    def preprocess(self, examples):
        inputs = [ex[self.arguments.source_id] for ex in examples["translation"]]
        targets = [ex[self.arguments.target_id] for ex in examples["translation"]]
        inputs = [self.arguments.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.arguments.max_source_length,
            padding=self.padding,
            truncation=True,
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.arguments.max_target_length,
                padding=self.padding,
                truncation=True,
            )
        if self.padding == "max_length" and self.arguments.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.arguments.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(
            decoded_preds, decoded_labels
        )
        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        result = {"bleu": result["score"]}
        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def generate_diverse(
        self,
        input_text,
        output_count=8,
        max_length=128,
        beam_width=4,
        num_beam_groups=2,
        device="cuda",
    ):
        self.model = self.model.to(device)
        model_inputs = self.tokenizer(
            [input_text],
            truncation="longest_first",
            padding="longest",
            return_tensors="pt",
        ).to(device)
        translated = self.model.generate(
            **model_inputs,
            max_length=int(max_length),
            num_return_sequences=output_count,
            # number of beams for beam search
            num_beams=int(output_count * beam_width),
            # number of groups to divide num_beams into in order to ensure diversity
            num_beam_groups=num_beam_groups,
            repetition_penalty=1.3,
            # higher the penalty, the more diverse are the outputs
            diversity_penalty=0.5,
            early_stopping=True,
        )
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        output_texts = [t.strip() for t in tgt_text]
        output_texts = list(set(output_texts))
        output_texts.sort()
        return output_texts

    def generate_single(self, input_text, max_length=128, device="cuda"):
        self.model = self.model.to(device)
        model_inputs = self.tokenizer(
            [input_text],
            truncation="longest_first",
            padding="longest",
            return_tensors="pt",
        ).to(device)
        translated = self.model.generate(
            **model_inputs, max_length=int(max_length), early_stopping=True
        )
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text[0].strip()

    def train(self, valid, tensorboard=False):
        column_names = self.raw_datasets["train"].column_names
        train_dataset = self.raw_datasets["train"]
        with self.arguments.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                self.preprocess,
                batched=True,
                num_proc=self.arguments.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.arguments.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if valid:
            eval_dataset = self.raw_datasets["valid"]
            with self.arguments.main_process_first(
                desc="validation dataset map pre-processing"
            ):
                eval_dataset = eval_dataset.map(
                    self.preprocess,
                    batched=True,
                    num_proc=self.arguments.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not self.arguments.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )

        # Data collator
        label_pad_token_id = (
            -100
            if self.arguments.ignore_pad_token_for_loss
            else self.tokenizer.pad_token_id
        )
        if self.arguments.pad_to_max_length:
            data_collator = default_data_collator
        else:
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=self.arguments.pad_to_multiple_of,
            )

        if not valid:
            self.arguments.validation_file = None
            self.arguments.evaluation_strategy = None

        # Initialize our Trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if valid else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
            if self.arguments.predict_with_generate
            else None,
        )
        if not tensorboard:
            self.trainer.remove_callback(transformers.integrations.TensorBoardCallback)

        # Training
        checkpoint = None
        if self.arguments.resume_from_checkpoint is not None:
            checkpoint = self.arguments.resume_from_checkpoint
        elif self.last_checkpoint is not None:
            checkpoint = self.last_checkpoint

        torch.cuda.empty_cache()
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        torch.cuda.empty_cache()

        self.trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

    def valid(self):
        max_length = self.arguments.max_target_length
        num_beams = self.arguments.generation_num_beams

        eval_dataset = self.raw_datasets["valid"]
        column_names = self.raw_datasets["valid"].column_names
        with self.arguments.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                self.preprocess,
                batched=True,
                num_proc=self.arguments.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.arguments.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

        torch.cuda.empty_cache()
        metrics = self.trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="valid"
        )
        torch.cuda.empty_cache()

        metrics["eval_samples"] = len(eval_dataset)
        self.trainer.log_metrics("valid", metrics)
        self.trainer.save_metrics("valid", metrics)
