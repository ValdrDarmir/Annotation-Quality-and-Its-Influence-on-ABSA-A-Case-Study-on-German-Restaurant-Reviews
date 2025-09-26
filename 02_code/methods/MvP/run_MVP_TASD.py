import subprocess
import sys
import os
import pandas as pd
import numpy as np
import datetime

###
# CV
###

LEARNING_RATE = 1e-4
DATA_PATH = '/home/niklasdonhauser/datasets'
OUTPUT_PATH = '/home/niklasdonhauser/MvP/results'
TASK = 'tasd'
SEED = 42
BATCH_SIZE = 16
GRADIENT_STEPS = 1
MODEL = 't5-base'
DATASET = "crowd/tasd" # "llm_ft_small_test_examples/tasd"
EVAL_TYPE = 'test'
BASE_EPOCHS = 15

for SEED in [5, 10, 15, 20, 25] :  #   [5, 10, 15, 20, 25]    
    for DATA_SETTING in ['orig']:
        command = f"CUDA_VISIBLE_DEVICES={int(0)} python3 src/classifier.py \
            --data_path {DATA_PATH} \
            --model_name_or_path {MODEL} \
            --dataset {DATASET} \
            --eval_type {EVAL_TYPE} \
            --data_setting {DATA_SETTING} \
            --output_dir {OUTPUT_PATH} \
            --num_train_epochs {BASE_EPOCHS} \
            --save_top_k 0 \
            --task {TASK} \
            --top_k 5 \
            --ctrl_token post \
            --multi_path \
            --num_path 5 \
            --seed {SEED} \
            --train_batch_size {BATCH_SIZE} \
            --gradient_accumulation_steps {GRADIENT_STEPS} \
            --learning_rate {LEARNING_RATE} \
            --sort_label \
            --data_ratio 1.0 \
            --check_val_every_n_epoch {BASE_EPOCHS + 1} \
            --agg_strategy vote \
            --eval_batch_size 16 \
            --constrained_decode"
            
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
            OUTPUT_PATH,
            f"{TASK}_{dataset_parts[0]}-{dataset_parts[1]}_{EVAL_TYPE}_{DATA_SETTING}-{DATA_SETTING[0]}_{LEARNING_RATE}_{BATCH_SIZE}_{BASE_EPOCHS}_{SEED}"
        )

        
        # Safety check: if log_folder doesn't exist, fallback to "results"
        if not os.path.exists(log_folder):
            print(f"⚠️ Log folder '{log_folder}' not found. Using 'results' instead.")
            log_folder = OUTPUT_PATH
        
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
        # Add the environment variable as a prefix
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = 'false'
        
