import os
import gc
import numpy as np
from . import utils
import torch
from datasets import load_dataset, load_metric
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
    

class Trainer:
    def __init__(self, arguments):
        self.arguments = arguments
        self.last_checkpoint = None
        if (
            os.path.isdir(self.arguments.output_dir)
            and not self.arguments.overwrite_output_dir
        ):
            self.last_checkpoint = get_last_checkpoint(
                self.arguments.output_dir)
            if (
                self.last_checkpoint is None
                and len(os.listdir(self.arguments.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({self.arguments.output_dir}) already exists and is not empty. "
                )
        set_seed(self.arguments.seed)
        self.metric = load_metric(self.arguments.metric)
        if not self.arguments.mode:
            self.arguments.mode = self.determine_model_type(
                self.arguments.model_name_or_path)
            if self.arguments.mode == "seq2seq":
                print("Training seq2seq Language Model")
            elif  self.arguments.mode == "clm":
                print("Training causal Language Model")
            else:
                print("Set TrainingArguments.mode to", self.arguments.mode)
        # squeeze out more RAM to load larger models
        gc.collect()
        self.load_model()
        self.is_prompt_tuning_only = False
        gc.collect()
        if self.arguments.torch_ort:
            from torch_ort import ORTModule
            self.model = ORTModule(self.model)
        try:
            self.load_dataset()
        except Exception as e:
            print(e, "- no dataset loaded")

    def determine_model_type(self, model_name):
        mn = model_name.lower()
        if "t5" in mn or "bart" in mn or "pegasus" in mn or "opus" in mn:
            return "seq2seq"
        elif "gpt" in mn:
            return "clm"
        else:
            print("Unable to determine model type for", model_name)
            print("Set TrainingArguments.mode as `seq2seq` or `clm` to override")
            raise NotImplementedError(
                "Unable to determine model type for training")

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
                extension = self.arguments.train_file.split(".")[-1]
                if extension == "txt":
                    print("Loading text dataset")
                    extension = "text"
                    self.arguments.keep_linebreaks = True
                data_files["train"] = self.arguments.train_file
            if self.arguments.validation_file is not None:
                data_files["valid"] = self.arguments.validation_file
            if self.arguments.test_file is not None:
                data_files["test"] = self.arguments.test_file
            self.raw_datasets = load_dataset(
                extension, data_files=data_files, cache_dir=self.arguments.cache_dir
            )
            if "valid" not in self.raw_datasets.keys() and self.arguments.validation_split > 0:
                print("Auto-split validation set from training set:",
                      self.arguments.validation_split, "%")
                self.raw_datasets["valid"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{self.arguments.validation_split}%]",
                    cache_dir=self.arguments.cache_dir,
                )
                self.raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{self.arguments.validation_split}%:]",
                    cache_dir=self.arguments.cache_dir,
                )
                
    def GradCheckpoint(self):
        self.arguments.gradient_checkpointing = True
        
    def FreezeEmbeds(self):
        self.freeze(embeddings=True)

    def freeze_params(self, module):
        for par in module.parameters():
            par.requires_grad = False

    def unfreeze_params(self, module):
        for par in module.parameters():
            par.requires_grad = True

    def freeze(self, encoder=None, decoder=None, embeddings=None):
        if encoder:
            self.freeze_params(self.model.encoder)
        elif encoder==False:
            self.unfreeze_params(self.model.encoder)
        if decoder:
            self.freeze_params(self.model.decoder)
        elif decoder==False:
            self.unfreeze_params(self.model.decoder)
        # embeddings:
        embedding_layers = []
        try:
            embedding_layers.append(self.model.shared)
        except Exception as e:
            print(e, "(probably not a encoder-decoder model)")
        try:
            embedding_layers.append(self.model.transformer.wte)
            embedding_layers.append(self.model.transformer.wpe)
        except Exception as e:
            print(e, "(probably not a GPT-Neo model)")
        for el in embedding_layers:
            if embeddings:
                self.freeze_params(el)
            elif embeddings==False:
                self.unfreeze_params(el)

    def resize_token_embeddings_layer(self, round_up=True):
        vocab_size = len(self.tokenizer)
        if round_up:
            vocab_size = utils.round_up(vocab_size, to=8)
        self.model.resize_token_embeddings(vocab_size)

    def add_new_tokens(self, additional_special_tokens):
        """List of additional special tokens to add"""
        special_tokens_dict = {
            "additional_special_tokens": additional_special_tokens}
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
            self.reinit_new_embeddings(
                self.model.shared, weight_indices=freeze_range)
            self.is_prompt_tuning_only = True
        else:
            raise NotImplementedError("Non-hook method not implemented")

    def count_parameters(self, trainable_only=False, return_M=False):
        if trainable_only:
            params = sum(p.numel()
                         for p in self.model.parameters() if p.requires_grad)
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
        if self.arguments.mode == "seq2seq":
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                self.arguments.model_name_or_path,
                from_tf=bool(".ckpt" in self.arguments.model_name_or_path),
                config=self.config,
                cache_dir=self.arguments.cache_dir,
            )
        elif self.arguments.mode == "clm":
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.arguments.model_name_or_path,
                from_tf=bool(".ckpt" in self.arguments.model_name_or_path),
                config=self.config,
                cache_dir=self.arguments.cache_dir,
            )
        else:
            print("Model name:", self.arguments.model_name_or_path)
            print("Training mode:", self.arguments.mode)
            NotImplementedError(
                "Unable to load this type of model for this training mode")
        if self.arguments.model_parallel_gpus > 1:
            print("Model parallel on", self.arguments.model_parallel_gpus, "GPU:")
            device_map = {}
            total_num_layers = self.config.num_layers
            layer_per_gpu = int(total_num_layers /
                                self.arguments.model_parallel_gpus)
            for i in range(self.arguments.model_parallel_gpus):
                layers = list(
                    range(i * layer_per_gpu, (i + 1) * layer_per_gpu))
                device_map[i] = layers
                print("* place layers", layers, "on GPU", i)
            self.model.parallelize(device_map)
        self.padding = "max_length" if self.arguments.pad_to_max_length else False

    def preprocess(self, examples):
        inputs = [ex[self.arguments.source_id]
                  for ex in examples["translation"]]
        targets = [ex[self.arguments.target_id]
                   for ex in examples["translation"]]
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
                [(l if l != self.tokenizer.pad_token_id else -100)
                 for l in label]
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
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True)
        if self.arguments.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels,
                              self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)
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
        tgt_text = self.tokenizer.batch_decode(
            translated, skip_special_tokens=True)
        output_texts = [t.strip() for t in tgt_text]
        output_texts = list(set(output_texts))
        output_texts.sort()
        return output_texts

    def generate_single(self, input_text, max_length=128, device="cuda"):
        self.model = self.model.to(device)
        model_inputs = self.tokenizer(
            [input_text],
            truncation="longest_first",
            return_tensors="pt",
        ).to(device)
        translated = self.model.generate(
            **model_inputs, max_length=int(max_length), early_stopping=True
        )
        tgt_text = self.tokenizer.batch_decode(
            translated, skip_special_tokens=True)
        return tgt_text[0].strip()

    def train(self, valid=False, batch_size=None, max_source_length=None, max_target_length=None, tensorboard=False, dev_mode=False, mode=None):
        gc.collect()
        if mode == None:
            mode = self.arguments.mode
        if mode == "seq2seq":
            self.train_seq2seq(valid=valid, batch_size=batch_size, max_source_length=max_source_length,
                               max_target_length=max_source_length, tensorboard=tensorboard, dev_mode=dev_mode)
        elif mode == "clm":
            self.train_clm(valid=valid, batch_size=batch_size, max_source_length=max_source_length,
                           max_target_length=max_source_length, tensorboard=tensorboard, dev_mode=dev_mode)
        else:
            raise NotImplementedError("Training mode not implemented")

    def train_clm(self, valid=False, batch_size=None, max_source_length=None, max_target_length=None, tensorboard=False, dev_mode=False):
        column_names = self.raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            output = self.tokenizer(examples[text_column_name])
            return output
        with self.arguments.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = self.raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.arguments.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.arguments.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        if self.arguments.block_size is None:
            block_size = self.tokenizer.model_max_length
        elif self.arguments.block_size > self.tokenizer.model_max_length:
            print("block_size > tokenizer.model_max_length, reduce to",
                  self.tokenizer.model_max_length)
            block_size = self.tokenizer.model_max_length
        else:
            block_size = int(self.arguments.block_size)

        def group_texts(examples):
            concatenated_examples = {
                k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size]
                    for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        with self.arguments.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=self.arguments.preprocessing_num_workers,
                load_from_cache_file=not self.arguments.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        train_dataset = lm_datasets["train"]
        if valid:
            eval_dataset = lm_datasets["valid"]

        if not valid:
            # override validation
            self.arguments.validation_file = None
            self.arguments.evaluation_strategy = None

        if batch_size:
            print("Override batch_size to", batch_size)
            self.arguments.per_device_train_batch_size = int(batch_size)
            self.arguments.per_device_eval_batch_size = int(batch_size)

        if dev_mode:
            print("Dev mode, setting max_steps to 10")
            self.arguments.max_steps = 10

        self.trainer = transformers.Trainer(
            model=self.model,
            args=self.arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if valid else None,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
        )
        if not tensorboard:
            self.trainer.remove_callback(
                transformers.integrations.TensorBoardCallback)

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

        return train_result

    def train_seq2seq(self, valid=False, batch_size=None, max_source_length=None, max_target_length=None, tensorboard=False, dev_mode=False):
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
            # override validation
            self.arguments.validation_file = None
            self.arguments.evaluation_strategy = None

        if batch_size:
            print("Override batch_size to", batch_size)
            self.arguments.per_device_train_batch_size = int(batch_size)
            self.arguments.per_device_eval_batch_size = int(batch_size)

        if max_source_length:
            print("Override max_source_length to", max_source_length)
            self.arguments.max_source_length = int(max_source_length)

        if max_target_length:
            print("Override max_target_length to", max_target_length)
            self.arguments.max_target_length = int(max_target_length)

        if dev_mode:
            print("Dev mode, setting max_steps to 10")
            self.arguments.max_steps = 10

        # Initialize our Trainer
        self.trainer = transformers.Seq2SeqTrainer(
            model=self.model,
            args=self.arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if valid else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
            if self.arguments.predict_with_generate
            else None,
            callbacks=[utils.L2FetchCallback],
            optimizers=[self.arguments.optimizer, None],
        )
        if not tensorboard:
            self.trainer.remove_callback(
                transformers.integrations.TensorBoardCallback)

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

    def valid_seq2seq(self):
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
