import subprocess
import os
import pandas as pd
import numpy as np
import sys
import datetime

RESULTS_PATH = '/home/niklasdonhauser/Paraphrase/results'
TASK = 'tasd'
BATCH_SIZE = 16
BASE_EPOCHS = 30
LEARNING_RATE = 3e-4
GRADIENT_STEPS = 1
DATA_PATH = '/home/niklasdonhauser/datasets'
SEED = 5
MODEL_NAME = 't5-base'
BASE_EPOCHS = 15
###
# Hyperparameter Validation Phase
###
DATASET = "crowd/tasd"
EVAL_TYPE = 'test'

for SEED in [5, 10, 15, 20, 25] : #[5,10,15,20,25]        
    for DATA_SETTING in ['orig']:

        command = f"CUDA_VISIBLE_DEVICES={int(0)} python3 classifier.py \
                --task {TASK} \
                --data_path {DATA_PATH} \
                --data_setting {DATA_SETTING} \
                --dataset {DATASET} \
                --learning_rate {LEARNING_RATE} \
                --per_device_train_batch_size {BATCH_SIZE} \
                --num_train_epochs {BASE_EPOCHS} \
                --model_name_or_path {MODEL_NAME} \
                --output_dir {RESULTS_PATH} \
                --gradient_accumulation_steps {GRADIENT_STEPS} \
                --seed {SEED} \
                --eval_type {EVAL_TYPE}"
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
            print(line, end='')       # print in real-time
            output_lines.append(line) # store for log
        
        process.wait()
        
        # Build log folder path
        dataset_parts = DATASET.split('/')
        log_folder = os.path.join(
            RESULTS_PATH,
            f"{TASK}_{dataset_parts[0]}-{dataset_parts[1]}_{EVAL_TYPE}_{DATA_SETTING}-{DATA_SETTING[0]}_{LEARNING_RATE}_{BATCH_SIZE}_{BASE_EPOCHS}_{SEED}"
        )

        
        # Safety check: if log_folder doesn't exist, fallback to "results"
        if not os.path.exists(log_folder):
            print(f"⚠️ Log folder '{log_folder}' not found. Using 'results' instead.")
            log_folder = RESULTS_PATH
        
        # Ensure folder exists
        print(log_folder)
        os.makedirs(log_folder, exist_ok=True)
        
        # Save all output to a log file after completion
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(
            log_folder, f"log_{TASK}_{DATASET.replace('/', '-')}_seed{SEED}_{timestamp}.txt"
        )
        
        with open(log_file, "w", encoding="utf-8") as f:
            f.writelines(output_lines)
        
        print(f"\n✅ Log saved to: {log_file}")