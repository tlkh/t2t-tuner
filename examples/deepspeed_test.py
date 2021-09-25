import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import time
import t2t

trainer_arguments = t2t.TrainerArguments(
    # model
    model_name_or_path="t5-3b",
    cache_dir="/workspace/cache",
    # data inputs
    train_file="../sample_data/trainlines.json",
    max_source_length=128,
    max_target_length=128,
    # taining outputs
    output_dir="/tmp/saved_model",
    overwrite_output_dir=True,
    # training settings
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=1e-5,
    prefix="predict sentiment: ",
    deepspeed="../ds_config/zero_3.json",
)
trainer = t2t.Trainer(arguments=trainer_arguments)

trainer.model_summary()

st = time.time()
trainer.train(valid=False)
et = time.time()

print("Time taken:", int(et-st), "seconds")

