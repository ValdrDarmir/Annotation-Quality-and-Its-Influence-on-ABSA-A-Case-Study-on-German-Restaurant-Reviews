import subprocess
import sys
import os
import pandas as pd
import numpy as np
import datetime

DATA_DIR = '/home/niklasdonhauser/datasets'
MODEL = 'GCN'
OUTPUT_PATH = '/home/niklasdonhauser/HIER-GCN/results'
BATCH_SIZE = 8
# SEED = 5
SEEDS = [5,10,15,20,25]
EPOCHS = 40 # 40
LEARNING_RATE = 5e-5
MODEL_NAME = 'deepset/gbert-base'
# experts/acsa
DATASET = "crowd/acsa"
TASK = 'acsa'
RESULTS_PATH = '/home/niklasdonhauser/HIER-GCN/results'


EVAL_TYPE = 'test'
for SEED in SEEDS:  
    for DATA_SETTING in ['orig']:
        SYS_EXEC = f"CUDA_VISIBLE_DEVICES={int(0)} python3 " if os.name == "posix" else "python " 
        command = f"{SYS_EXEC} run_classifier_gcn.py \
            --seed {SEED} \
            --task {TASK} \
            --dataset {DATASET} \
            --eval_type {EVAL_TYPE} \
            --data_setting {DATA_SETTING} \
            --model_name_or_path {MODEL_NAME} \
            --model_type {MODEL}\
            --do_lower_case \
            --data_dir {DATA_DIR} \
            --max_seq_length 128 \
            --per_device_train_batch_size {BATCH_SIZE} \
            --learning_rate {LEARNING_RATE} \
            --num_train_epochs {EPOCHS} \
            --output_dir {OUTPUT_PATH}"

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
            f"{dataset_parts[1]}_{EVAL_TYPE}_{DATA_SETTING}-{DATA_SETTING[0]}_{LEARNING_RATE}_{BATCH_SIZE}_{EPOCHS}_{SEED}"
        )
        # Save all output to a log file after completion
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(
            log_folder, f"log_{TASK}_{DATASET.replace('/', '-')}_seed{SEED}_{timestamp}.txt"
        )
        with open(log_file, "w", encoding="utf-8") as f:
            f.writelines(output_lines)

        print(f"\nLog saved to: {log_file}")