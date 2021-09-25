import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments


@dataclass
class TrainerArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(default="./saved_model/")
    metric: Optional[str] = field(default="sacrebleu")
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    block_size: Optional[int] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    validation_split: Optional[int] = field(default=0)
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
    mode: Optional[str] = field(default=None)
    optimizer: Optional[torch.optim.Optimizer] = field(default=None)
    torch_ort: Optional[bool] = field(default=False)
