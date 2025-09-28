import os
import time
import sys
from llm import LLM
from dataloader import DataLoader, ExamplesLoader
from promptloader import PromptLoader
import concurrent.futures
from validator import validate_label
import json
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# llm_ft_small_test_examples
path = "/home/niklasdonhauser/datasets/crowd/acsa"
dataloader = DataLoader(base_path=path)
examples_loader = ExamplesLoader(
    "/home/niklasdonhauser/LLM-Few-Shot-OLAMA/fs_example/acsa_crowd_50_shots.jsonl")
promptloader = PromptLoader()

def run_with_timeout(llm, prompt, seed, timeout=300, max_retries=1):
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future = executor.submit(llm.predict, prompt, seed)
            try:
                output, duration = future.result(timeout=timeout)
                return output, duration
            except concurrent.futures.TimeoutError:
                future.cancel()  # won’t actually stop the process, but keeps logic consistent
                print(f"[Attempt {attempt}] Prediction timed out. Retrying...")
            except Exception as e:
                print(f"[Attempt {attempt}] Prediction failed with exception: {e}. Retrying...")

    # After max_retries
    raise TimeoutError(f"Prediction failed {max_retries} times. Moving on.")
# ---------------------------------------------------------------------------------------------
def zero_shot(TASK, DATASET_NAME, LLM_BASE_MODEL, SEED, N_FEW_SHOT, dataset_name):

    print("TASK:", TASK)
    print(f"DATASET_NAME: {DATASET_NAME}")
    print(f"LLM_BASE_MODEL: {LLM_BASE_MODEL}")
    print(f"SEED: {SEED}")
    print(f"N_FEW_SHOT: {N_FEW_SHOT}")

    # Load Model

    llm = LLM(
        base_model=LLM_BASE_MODEL,
        parameters=[
            {"name": "stop", "value": [")]"]},
            {"name": "num_ctx", "value": 4096}
        ]
    )
    # Load Eval Dataset
    dataset_train = dataloader.load_data("train", name=dataset_name)
    # Unique Aspect Categories

    # unique_aspect_categories = sorted(
    #     {aspect['aspect_category'] for entry in dataloader.load_data("test") for aspect in entry['aspects']})
    # unique_aspect_categories = sorted(unique_aspect_categories)
    unique_aspect_categories = ['ambiente',
                                'essen', 'gesamteindruck', 'preis', 'service']
    print("Unique Aspect Categories:", unique_aspect_categories)
    predictions = []

    #############################################
    #############################################

    # Load Train Dataset
    # fs_embeddings = get_embeddings_dict([example["text"] for example in few_shot_split_0])

    # label
    for idx, example in enumerate(dataset_train):
        prediction = {
            "task": TASK,
            "dataset_name": DATASET_NAME,
            "dataset_type": "test",
            "llm_base_model": LLM_BASE_MODEL,
            "id": example["id"],
            "invalid_precitions_label": [],
            "init_seed": SEED,
        }

        seed = SEED
        few_shot_examples = examples_loader.load_examples()
        prompt = promptloader.load_prompt(task=TASK,
                                          aspects=unique_aspect_categories,
                                          examples=few_shot_examples,
                                          seed_examples=seed,
                                          # use False if specific order
                                          input_example=example)
        print(f"prompt: {prompt} \n\n")
        correct_output = False
        max_inner_retries = 1 #9
        inner_attempt = 0
        
        while not correct_output and inner_attempt < max_inner_retries:
            inner_attempt += 1
            try:
                print(f"try example {idx} \n")
                output, duration = run_with_timeout(llm, prompt, seed)
            except TimeoutError as e:
                print(f"[Example {idx}] Timeout on attempt {inner_attempt}: {e}")
                continue  # retry up to max_inner_retries
            except Exception as e:
                print(f"[Example {idx}] Other error on attempt {inner_attempt}: {e}")
                continue
        
            output_raw = output
            print("output_raw:", output_raw)
        
            # delete new lines
            output = output.replace("\n", "")
        
            validator_output = validate_label(
                output, example["text"], unique_aspect_categories, task=TASK, allow_small_variations=True)
            print("validator_output:", validator_output)
        
            if validator_output[0] is not False:
                prediction["pred_raw"] = output_raw
                prediction["pred_label"] = validator_output[0]
                prediction["duration_label"] = duration
                prediction["seed"] = seed
                correct_output = True
            else:
                prediction["invalid_precitions_label"].append(
                    {"pred_label_raw": output_raw, "pred_label": validator_output[0],
                     "duration_label": duration, "seed": seed, "regeneration_reason": validator_output[1]}
                )
                seed += 5
        
        # If we never got a correct output
        if not correct_output:
            prediction["pred_label"] = []
            prediction["duration_label"] = duration
            prediction["seed"] = seed

        print("########## ", idx, "\nText:", example["text"], "\nLabel:",
              prediction["pred_label"], "\nRegenerations:", prediction["invalid_precitions_label"])
        predictions.append(dict(prediction, **example))

    dir_path = "/home/niklasdonhauser/LLM-Few-Shot-OLAMA/result"

    # Create the directories if they don't exist
    os.makedirs(dir_path, exist_ok=True)
    safe_model_name = LLM_BASE_MODEL.replace(":", "_").replace("/", "_")
    # match = re.search(r"_(\d+)\.jsonl$", dataset_name)
    # number = int(match.group(1))
    change_name = f"{task}_{dataset}_{safe_model_name}_{seed}_{fs}"
    with open(f"{dir_path}/{change_name}.json",
              'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------------------------
seeds = [10, 20] # 5, 10, 15, 20, 25 
fs = 50
dataset = "crowd"
task = "acsa"
# task = "acsa"
model = "gemma3:27b"
dataset_names = ["test.json"]

for seed in seeds:
    for dataset_name in dataset_names:
        # match = re.search(r"_(\d+)\.jsonl$", dataset_name)
        # number = int(match.group(1))
        safe_model_name = model.replace(":", "_").replace("/", "_")
        change_name = f"{task}_{dataset}_{safe_model_name}_{seed}_{fs}"
        file_path = f"/home/niklasdonhauser/LLM-Few-Shot-OLAMA/result/{change_name}.json"
        print(file_path)
        # Prüfen, ob die Datei bereits existiert
        if not os.path.exists(file_path):
            time.sleep(2)
            print("not exist")
            zero_shot(task, dataset,
                      model, seed, fs, dataset_name)
        else:
            print(f"Skipping: {file_path} already exists.")
