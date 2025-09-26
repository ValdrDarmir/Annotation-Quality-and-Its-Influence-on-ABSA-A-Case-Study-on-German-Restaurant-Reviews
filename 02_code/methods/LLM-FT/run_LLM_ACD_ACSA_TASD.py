import subprocess
import sys
import os
import pandas as pd
import numpy as np
import datetime

# CONSTANTS

# MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
MODEL_NAME = "meta-llama/Llama-3.1-8B"
LORA_DROPOUT = 0.05
QLORA_QUANT = 4
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LR_SCHEDULER = 'linear'
LORA_R, LORA_ALPHA = 64, 16
LEARNING_RATE = 2e-4
OUTPUT_DIR = '/home/niklasdonhauser/LLM-Fine-Tune/results'
DATA_SETTING = 'orig'

###
# Hyperparameter Validation Phase
###

DATASET = "crowd/acsa"
EVAL_TYPE = 'test'
# different datasets, so dont use two task in a loop
# exchange the check for acsa task in train.py
for TASK in ['acsa']: # acd tasd
    for SEED in [5, 10, 15, 20, 25] :   #[5, 10, 15, 20, 25]  [5]   
        for DATA_SETTING in ['orig']:
            if TASK == 'acsa':
                EPOCHS = 5
            elif TASK == 'tasd':
                EPOCHS = 6
            else:
                EPOCHS = 1
            # --bf16 \
            command = f"CUDA_VISIBLE_DEVICES={int(0)} python3 train.py \
            --model_name_or_path {MODEL_NAME} \
            --lora_r {LORA_R} \
            --lora_alpha {LORA_ALPHA} \
            --lora_dropout {LORA_DROPOUT} \
            --quant {QLORA_QUANT} \
            --eval_type {EVAL_TYPE} \
            --learning_rate {LEARNING_RATE} \
            --per_device_train_batch_size {BATCH_SIZE} \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {GRADIENT_ACCUMULATION_STEPS} \
            --num_train_epochs {EPOCHS} \
            --dataset {DATASET} \
            --data_setting {DATA_SETTING} \
            --task {TASK} \
            --lr_scheduler {LR_SCHEDULER} \
            --fp16 \
            --group_by_length \
            --output_dir {OUTPUT_DIR} \
            --seed {SEED} \
            --flash_attention"
            
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
            dataset_parts = DATASET.split('/')  # ['experts', 'acsa']
            log_folder = os.path.join(
                OUTPUT_DIR,
                f"{TASK}_{dataset_parts[0]}-{dataset_parts[1]}_{EVAL_TYPE}_{DATA_SETTING}-{DATA_SETTING[0]}_{LEARNING_RATE}_{BATCH_SIZE}_{EPOCHS}_{SEED}"
            )

            
            # Safety check: if log_folder doesn't exist, fallback to "results"
            if not os.path.exists(log_folder):
                print(f"⚠️ Log folder '{log_folder}' not found. Using 'results' instead.")
                log_folder = OUTPUT_DIR
            
            # Ensure folder exists
            os.makedirs(log_folder, exist_ok=True)
            
            # Save all output to a log file after completion
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file = os.path.join(
                log_folder, f"log_{TASK}_{DATASET.replace('/', '-')}_seed{SEED}_{timestamp}.txt"
            )
            
            with open(log_file, "w", encoding="utf-8") as f:
                f.writelines(output_lines)
            
            print(f"\n✅ Log saved to: {log_file}")