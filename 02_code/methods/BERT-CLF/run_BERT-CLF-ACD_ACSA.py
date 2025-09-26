import subprocess
import os
import pandas as pd
import numpy as np
import datetime

# Config
DATA_PATH = "/home/niklasdonhauser/datasets"
OUTPUT_PATH = "/home/niklasdonhauser/BERT-CLF/results"
MODEL_NAME = "deepset/gbert-base"
TASK = "acsa"
# experts/acsa
DATASET = "crowd/acsa"
EVAL_TYPE = "test"
DATA_SETTING = "orig"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
SEEDS = [5, 10, 15, 20, 25]
# SEEDS = [5]
BASE_EPOCHS = 40  #40

for SEED in SEEDS:
    SYS_EXEC = f"CUDA_VISIBLE_DEVICES={int(0)} python3 " if os.name == "posix" else "python "

    command = f"{SYS_EXEC} classifier.py \
        --seed {SEED} \
        --task {TASK} \
        --dataset {DATASET} \
        --eval_type {EVAL_TYPE} \
        --data_setting {DATA_SETTING} \
        --model_name_or_path {MODEL_NAME} \
        --learning_rate {LEARNING_RATE} \
        --per_device_train_batch_size {BATCH_SIZE} \
        --num_train_epochs {BASE_EPOCHS} \
        --output_dir {OUTPUT_PATH} \
        --data_path {DATA_PATH}"

    print(f"Running command:\n{command}")

    # Run process and capture output line by line
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    output_lines = []
    for line in process.stdout:
        print(line, end='')      # print in real-time
        output_lines.append(line) # store for log

    process.wait()
    dataset_parts = DATASET.split('/')  # ['experts', 'acsa']
    log_folder = os.path.join(
        OUTPUT_PATH,
        f"{TASK}_{dataset_parts[0]}",      # first part as main folder
        f"{dataset_parts[1]}_{EVAL_TYPE}_{DATA_SETTING}-{DATA_SETTING[0]}_{LEARNING_RATE}_{BATCH_SIZE}_{BASE_EPOCHS}_{SEED}"
    )
    # Save all output to a log file after completion
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(
        log_folder, f"log_{TASK}_{DATASET.replace('/', '-')}_seed{SEED}_{timestamp}.txt"
    )
    with open(log_file, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    print(f"\nLog saved to: {log_file}")