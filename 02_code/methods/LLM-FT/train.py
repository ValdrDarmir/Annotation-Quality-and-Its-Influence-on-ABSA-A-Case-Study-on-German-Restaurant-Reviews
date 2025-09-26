import unsloth
import pandas as pd
import random
import bitsandbytes as bnb
import torch
import transformers
import numpy as np
import os
import time
import json
import sys
import re

utils = os.path.abspath('/home/niklasdonhauser/utils') # Relative path to utils scripts
sys.path.append(utils)

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from tqdm import tqdm
from datetime import datetime, timedelta
from helpers_llm import *
from transformers import set_seed
from preprocessing import loadDataset, splitForEvalSetting
from evaluation import createResults
from unsloth import FastLlamaModel
from transformers import StoppingCriteria, StoppingCriteriaList
from ast import literal_eval
from argparse import ArgumentParser

HF_TOKEN = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REGEX_ASPECTS_ACSD = r"\(\'([^\']+)\',\s*([^,]+),\s*([^\']+)\)"

def safe_recursive_pattern(depth, max_depth):
    quoted_content = r'"(?:[^"\\]|\\.)*"'
    
    if depth >= max_depth:
        return rf'(?:{quoted_content}|[^()])*'
    return rf'\((?:{quoted_content}|[^()]|{safe_recursive_pattern(depth + 1, max_depth)})*\)'
    
def convertLabels(labels, task, label_space):
    false_predictions = []
    conv_l = []
    label_space = sorted(set(lab.split(':')[0] for lab in label_space)) if task == 'acd' else label_space
    for sample in labels:
        conv_s = []
        if sample:
            for pair in sample:
                if task != 'acd':
                    pair_str = ':'.join(label.replace('"', '').replace("'", "") for label in pair[:2])
                else:
                    pair_str = pair
                    
                if pair_str in label_space or task == 'e2e' or task == 'e2e-e':
                    conv_s.append(':'.join([pair_str, pair[2]]) if task == 'tasd' else pair_str)
                else:
                    false_predictions.append(pair_str)
        conv_l.append(conv_s)

    return conv_l, false_predictions
def extractAspects(output, task):
    
    result = []
    if task == 'tasd':
        max_depth = 5
        pattern_targets = re.compile(safe_recursive_pattern(0, max_depth))
        pairs = pattern_targets.findall(output)

        for pair in pairs:
            try: 
                match = literal_eval(pair)
                result.append([match[1], match[2], match[0]])
            except:
                pass
    
    elif task == 'acd':
        REGEX_ASPECTS_ACD = r"\[([^\]]+)\]"
        pattern_asp = re.compile(REGEX_ASPECTS_ACD)
        matches = pattern_asp.findall(output)

        for match in matches:
            aspects = [s.strip().strip("'\"") for s in match.split(',')]
            result.extend(aspects)

    elif task == 'acsa':
        REGEX_ASPECTS_ACSA = r"\(([^,]+),\s*([^,]+)\)"
        pattern_pairs = re.compile(REGEX_ASPECTS_ACSA)
        
        pairs = pattern_pairs.findall(output)

        for pair in pairs:
            result.append([pair[0], pair[1]])
        
    return result

        
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            stop_len = stop.shape[-1]
            input_len = input_ids.shape[1]
            if input_len >= stop_len:                
                if torch.all(input_ids[0, -stop_len:] == stop):
                    return True
        return False

def find_all_linear_names(model, bits):
    """
    Identifies the names of linear modules within the model that should be adapted using LoRA.

    Args:
        model (torch.nn.Module): The transformer model.
        bits (int): Bit precision for quantized layers (4 or 8).

    Returns:
        List[str]: Names of modules to apply LoRA on.
    """
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def createModel(config):
    """
    Loads a pretrained model and tokenizer with LoRA adaptation.

    Args:
        config (Config): Model and LoRA configuration parameters.

    Returns:
        Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]: LoRA-adapted model and tokenizer.
    """
    # compute_dtype = torch.bfloat16 if config.bf16 else torch.float32
    if config.bf16:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
    print(f'Loading model from {config.model_name_or_path}...')
    model, tokenizer = FastLlamaModel.from_pretrained(
        config.model_name_or_path,
        max_seq_length=config.max_seq_length,
        dtype=compute_dtype,
        load_in_4bit=True if config.quant == 4 else False
    )
    # tokenizer.add_special_tokens({'pad_token': tokenizer.convert_ids_to_tokens(model.config.eos_token_id)})
    lora_layers = find_all_linear_names(model, config.quant)
    print("\n\nFine-Tuning Lora Layers: ", lora_layers, "\n\n")
    model = FastLlamaModel.get_peft_model(
        model,
        target_modules=lora_layers,
        lora_alpha=config.lora_alpha,
        r=config.lora_r,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=config.seed
    )

    return model, tokenizer

def train(config, model, tokenizer, ds_train):
    """
    Trains the LoRA-adapted model using supervised fine-tuning.

    Args:
        config (Config): Training configuration parameters.
        model (torch.nn.Module): The adapted model.
        tokenizer (transformers.PreTrainedTokenizer): Associated tokenizer.
        ds_train (datasets.Dataset): The training dataset.

    Returns:
        Tuple[Dict, TrainingArguments]: Training stats and the used training arguments.
    """
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)

    ds_train = ds_train.map(lambda x: {'length': len(x['text'])})
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds_train,
        tokenizer=tokenizer,
        packing=False,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=TrainingArguments(
            bf16=config.bf16,
            fp16=config.fp16,
            ddp_find_unused_parameters=False,
            output_dir=f'outputs/{config.task}_{config.dataset}_{config.eval_type}_{config.num_train_epochs}_{config.seed}',
            seed=config.seed,
            report_to="none",
            disable_tqdm=False,
            max_grad_norm=0.3,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            group_by_length=config.group_by_length,
            num_train_epochs=config.num_train_epochs,
            logging_steps=config.logging_steps,
            save_strategy=config.save_strategy,
            optim=config.optim
        ),
    )

    time_start = time.time()
    train_result = trainer.train()
    time_end = time.time()
    training_time = str(timedelta(seconds=(time_end - time_start)))

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)

    trainer_stats = {
        "model_name": config.model_name_or_path,
        "task": config.task,
        "data_setting": config.data_setting,
        "dataset": config.dataset,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "eval_type": config.eval_type,
        "lr_scheduler_type": config.lr_scheduler_type,
        "seed": config.seed,
        "used_memory": used_memory,
        "used_memory_for_lora": used_memory_for_lora,
        "temperature": config.temperature, 
        "max_tokens": config.max_new_tokens,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "training_time": training_time,
    }

    # model.save_pretrained_merged('local/save_model', tokenizer, save_method = "merged_16bit",)  # Modell speichern
    
    return trainer_stats

def savePredictions(result, preds, golds, config, training_stats, eval_type):
    """
    Saves predictions, metrics, and configuration details to disk.

    Args:
        result (List[Dict]): Computed evaluation metrics.
        preds (List): Model predictions.
        golds (List): Ground truth labels.
        config (Config): Configuration object.
        training_args (TrainingArguments): Huggingface training args.
        eval_type (str): Evaluation dataset suffix (e.g., 'orig', 'dia').
    """

    output_path = f"{config.output_dir}/{config.task}_{config.dataset.replace('/', '-')}_{config.eval_type}_{config.data_setting}-{eval_type[0]}_{round(config.learning_rate,9)}_{config.per_device_train_batch_size}_{config.num_train_epochs}_{config.seed}/"
    os.makedirs(output_path, exist_ok=True)

    # Save individual metric outputs
    if config.task == 'acd':
        TASKS = ["asp"]
    elif config.task == 'acsa':
        TASKS = ["asp", "asp_pol", "pairs", "pol"]
    elif config.task == 'tasd':
        TASKS = ["asp", "asp_pol", "pairs", "pol", "phrases"]

    for idx, name in enumerate(TASKS):
        pd.DataFrame.from_dict(result[idx]).transpose().to_csv(f"{output_path}metrics_{name}.tsv", sep="\t")

    try:
        # Save predictions and gold labels for further analysis
        matched_samples = [
            {"predictions": pred, "gold_labels": gold}
            for pred, gold in zip(preds, golds)
        ]
        with open(os.path.join(output_path, 'predictions.json'), "w", encoding="utf-8") as f:
            json.dump({"test": matched_samples}, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save predictions. {e}")

    # Save config used for this run
    with open(os.path.join(output_path, 'config.json'), "w", encoding="utf-8") as f:
        json.dump(training_stats, f, indent=4, ensure_ascii=False)


def evaluate(config, model, tokenizer, prompts_test, ground_truth, label_space):
    """
    Runs evaluation by generating predictions and computing metrics.

    Args:
        config (Config): Configuration with evaluation parameters.
        model (torch.nn.Module): Trained model.
        prompts_test (Dataset): Dataset containing test prompts.
        gold_labels (List): Ground truth labels.
        label_space (List): Set of valid label classes.
        sampling_params (SamplingParams): Generation parameters.

    Returns:
        Tuple[List[Dict], List, List]: Evaluation results, gold labels, and model outputs.
    """
    FastLlamaModel.for_inference(model)  # Enable inference mode

    if config.dataset == 'crowd/acsa': # change dataset!!!!!!!
        if config.task == 'acsa':
            ground_truth = [[(asp, pol) for asp, pol in labels] for labels in ground_truth]
        elif config.task == 'acd':
            ground_truth = [[asp for asp, pol in labels] for labels in ground_truth]
    else:
        if config.task == 'tasd':
            ground_truth = [[(asp, pol, phr.lower() if phr == 'NULL' else phr) for asp, pol, phr in labels] for labels in ground_truth]
        elif config.task == 'acsa':
            ground_truth = [[(asp, pol) for asp, pol, phr in labels] for labels in ground_truth]
        elif config.task == 'acd':
            ground_truth = [[asp for asp, pol, phr in labels] for labels in ground_truth]
    
    inputs = tokenizer(prompts_test, return_tensors="pt", truncation=True, max_length=2048, padding= True)

    outputs = []

    ids = [torch.tensor([2595]), torch.tensor([128001]), torch.tensor([50724])] # ]\n\n End-of-Text ]\n
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=ids)])

    time_start = time.time()
    for i in tqdm(range(0, len(inputs['input_ids']), config.per_device_eval_batch_size)): # Eval_Batch_Size has to be 1 for early stopping
    
        model_outputs = model.generate(
            input_ids= inputs['input_ids'][i*config.per_device_eval_batch_size: i*config.per_device_eval_batch_size + config.per_device_eval_batch_size].to('cuda'), 
            attention_mask = inputs['attention_mask'][i*config.per_device_eval_batch_size: i*config.per_device_eval_batch_size + config.per_device_eval_batch_size].to('cuda'), 
            temperature = config.temperature, 
            do_sample = False,
            max_new_tokens = config.max_new_tokens,
            top_k = config.top_k,
            top_p = config.top_p, 
            stopping_criteria=stopping_criteria,
            use_cache = True)

        outputs.extend(tokenizer.batch_decode([model_outputs[0][len(inputs['input_ids'][0]):]], skip_special_tokens=True))
    
    time_end = time.time()

    print("Outputs: \n", outputs[:5])
    # Extract ABSA information from ground truth and model output
    predictions = [extractAspects(out, config.task) for out in outputs]

    print("Extracted Predictions: \n", predictions[:5])
    
    # Convert predictions and gold labels into evaluation format
    gold_labels, _ = convertLabels(ground_truth, config.task, label_space)
    pred_labels, false_predictions = convertLabels(predictions, config.task, label_space)

    print("Converted Labels: \n", pred_labels[:5])
    
    # Compute evaluation metrics
    results = createResults(pred_labels, gold_labels, label_space, config.task)
    return results, ground_truth, [{"output": out, "predictions": pred} for out, pred in zip(outputs, predictions)], time_end - time_start

def createPrompts(task, dataset, labels, train, test, few_shots, eos_token):
    PROMPT_TEMPLATE = globals()[f"PROMPT_{task.upper()}"]

    prompts_train = []
    prompts_test = []
    ground_truth = []
    
    few_shot_text = ''
    if few_shots > 0:
        
        few_shots = train[['text', 'labels']].sample(n=few_shots, random_state=42)
        
        for index, row in few_shots.iterrows():
            if dataset == 'crowd/acsa':  # update ND
                if task == 'acsa':
                    new_labels = [(asp, pol) for asp, pol in row['labels']]
                else:
                    new_labels = [asp for asp, pol in row['labels']]
            else:
                if task == 'tasd':
                    new_labels = [(phr.lower() if phr == 'NULL' else phr, asp, pol) for asp, pol, phr in row['labels']]
                elif task == 'acsa':
                    new_labels = [(asp, pol) for asp, pol, phr in row['labels']]
                else:
                    new_labels = [asp for asp, pol, phr in row['labels']]
            few_shot_text += f"Text: {row['text']}\nSentiment-Elemente: {str(new_labels)}\n\n"

    for index, row in train.iterrows():
        if dataset == 'crowd/acsa': # update for each dataset later ND
            if task == 'acsa':
                new_labels = [(asp, pol) for asp, pol in row['labels']]
            else:
                new_labels = [asp for asp, pol in row['labels']]
        else:
            if task == 'tasd':
                new_labels = [(phr.lower() if phr == 'NULL' else phr, asp, pol) for asp, pol, phr in row['labels']]
            elif task == 'acsa':
                print(row['labels'])
                new_labels = [(asp, pol) for asp, pol, phr in row['labels']]
            else:
                new_labels = [asp for asp, pol, phr in row['labels']]
        prompt = PROMPT_TEMPLATE.format(categories = labels, examples=few_shot_text)
        prompt += f"Text: {row['text']}\nSentiment-Elemente: {str(new_labels)}" + eos_token
        
        prompts_train.append(prompt)

    for index, row in test.iterrows():
        prompt = PROMPT_TEMPLATE.format(categories = labels, examples=few_shot_text)
        prompt += f"Text: {row['text']}\nSentiment-Elemente: "
    
        prompts_test.append(prompt)
        ground_truth.append(row['labels'])

    return prompts_train, prompts_test, ground_truth
    
def main():
    """
    Main entry point: loads configuration, prepares data, trains model, evaluates and saves results.
    """
    config = Config()
    print(', '.join("%s: %s" % item for item in vars(config).items()))

    set_seed(config.seed)

    model_save_path = 'local/save_model'
    
    if os.path.exists(model_save_path):
        print("Loading pre-trained model...")
        model, tokenizer = FastLlamaModel.from_pretrained(model_save_path,
                                                             max_seq_length = config.max_seq_length,
                                                            dtype = torch.bfloat16 if config.bf16 else torch.float32,
                                                            load_in_4bit = True if config.quant == 4 else False,
                                            )
        df_train, df_test, label_space = splitForEvalSetting(loadDataset(config.data_path, config.dataset), config.eval_type)
        categories = list(set([label.split(':')[0] for label in label_space]))
        prompts_train, prompts_test, ground_truth_labels = createPrompts(config.task, config.dataset, categories, df_train, df_test, 0, eos_token=tokenizer.eos_token)
        
    else:
    # Load model and tokenizer
        model, tokenizer = createModel(config)
    
        # Prepare datasets and prompts
        df_train, df_test, label_space = splitForEvalSetting(loadDataset(config.data_path, config.dataset), config.eval_type)
        categories = list(set([label.split(':')[0] for label in label_space]))
        prompts_train, prompts_test, ground_truth_labels = createPrompts(config.task, config.dataset, categories, df_train, df_test, 0, eos_token=tokenizer.eos_token)
        print(prompts_train[0])

        with open(f'./{config.task}_{config.num_train_epochs}_{config.seed}_prompt_train.txt', 'w') as f:
            f.write(f"{prompts_train[0].encode('utf-8')}\n\n")
            f.write(f"{prompts_train[1].encode('utf-8')}\n\n")
            f.write(f"{prompts_train[2].encode('utf-8')}\n\n")
            f.write(f"{prompts_train[3].encode('utf-8')}\n\n")
            f.write(f"{prompts_train[4].encode('utf-8')}\n\n")
            
        with open(f'./{config.task}_{config.num_train_epochs}_{config.seed}_prompt_test.txt', 'w') as f:
            f.write(f"{prompts_test[0].encode('utf-8')}\n\n")
            f.write(f"{prompts_test[1].encode('utf-8')}\n\n")
            f.write(f"{prompts_test[2].encode('utf-8')}\n\n")
            f.write(f"{prompts_test[3].encode('utf-8')}\n\n")
            f.write(f"{prompts_test[4].encode('utf-8')}\n\n")
            
        ds_train = Dataset.from_pandas(pd.DataFrame(prompts_train, columns=['text']))

        tokenizer
        
        # Train model
        training_stats = train(config, model, tokenizer, ds_train)

    # Generate evaluation samples
    results, all_labels, all_preds, eval_time = evaluate(config, model, tokenizer, prompts_test, ground_truth_labels, label_space)
    print(results)
    training_stats.update({"eval_time": eval_time})
    savePredictions(results, all_preds, all_labels, config, training_stats, 'orig')

    # Optionally evaluate on dialectal test split (if applicable)
    if 'test' in config.eval_type and config.dataset == 'transport':
        _, df_test_dia, _ = splitForEvalSetting(loadDataset(config.data_path, config.dataset, 'dia'), config.eval_type)
        _, prompts_test_dia, ground_truth_labels_dia = createPrompts(config.task, config.dataset, categories, df_train, df_test_dia, 0, eos_token=tokenizer.eos_token)
        results, all_labels, all_preds, eval_time = evaluate(config, model, tokenizer, prompts_test_dia, ground_truth_labels_dia, label_space)
        training_stats.update({"eval_time": eval_time})
        savePredictions(results, all_preds, all_labels, config, training_stats, 'dia')

class Config(object):
    def __init__(self):

        # General Params
        self.task = None
        self.output_dir = '/home/niklasdonhauser/LLM-Fine-Tune/results'
        self.data_path = '/home/niklasdonhauser/datasets'
        
        # Dataset Params
        self.lang = 'en'
        self.data_setting = 'orig'
        self.eval_type = 'dev'
        self.dataset = 'restaurant'

        ## LLM only
        # Training Params
        self.flash_attention = False
        self.quant = 4
        self.per_device_train_batch_size = 8
        self.gradient_accumulation_steps = 1
        self.learning_rate = None
        self.lr_scheduler_type = 'constant'
        self.num_train_epochs = None 
        self.group_by_length = True
        self.logging_steps = 50
        self.save_strategy = 'epoch'
        self.evaluation_strategy = 'epoch'
        self.optim = 'paged_adamw_32bit'
        self.lora_r = None
        self.lora_alpha = None
        self.lora_dropout = 0.05
        self.from_checkpoint = False
        
        # Inference Params
        self.epoch = None
        self.per_device_eval_batch_size = 8
        self.max_new_tokens = 200
        self.top_k = -1
        self.top_p = 1
        self.temperature = 0
        self.few_shots = None
        
        # Model Params
        self.model_name_or_path = "meta-llama/Meta-Llama-3-8B"
        self.seed = 42
        self.bf16 = False
        self.max_seq_length = 2048
        
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.update_config_with_args()

    def update_config_with_args(self):
        args = self.parser.parse_args()  # Behalte das Namespace-Objekt
        for key, value in vars(args).items():  # Iteriere über die Attribute des Namespace-Objekts
            if value is not None:  # Überschreibe nur, wenn ein Wert angegeben ist
                setattr(self, key, value)

        self.args = None
                
    def setup_parser(self):

        parser = ArgumentParser()

        # Model-related arguments
        parser.add_argument('--model_name_or_path', type=str, help='Base model name for training or inference')
        parser.add_argument('--seed', type=int, help='Seed to ensure reproducability.')
        parser.add_argument('--quant', type=int, help="How many bits to use for quantization.")
        # parser.add_argument('--bf16', action='store_true', help="Compute dtype of the model (uses bf16 if set).")
        parser.add_argument('--fp16', action='store_true', help="Use fp16 training.")
        parser.add_argument('--flash_attention', action='store_true', help='If to enable flash attention.')
        parser.add_argument('--epoch', type=int, help='Epoch checkpoint of the model.')
        parser.add_argument('--output_dir', type=str, help='Relative path to output directory.')
        parser.add_argument('--data_path', type=str, help='Relative path to data directory.')
        # Dataset-related arguments
        parser.add_argument('--dataset', type=str)
        parser.add_argument('--shots', type=str, help='Amount and style of few shot examples for evaluation.')
        parser.add_argument('--data_setting', type=str, required=True, help='Dataset setting, original or balanced.')
        parser.add_argument('--max_seq_length', type=int, help="Maximum context length during training and inference.")
        parser.add_argument('--task', type=str, default="acsa", help="Which ABSA Task the model was trained on. ['acd', 'acsa', 'acsd']")
        parser.add_argument('--eval_type', type=str, default="test", help="Which test set to choose. ['dev', 'test']")
        
         # Training arguments
        parser.add_argument('--per_device_train_batch_size', type=int, help='The training batch size per GPU.')
        parser.add_argument('--per_device_eval_batch_size', type=int, help='The evaluation batch size per GPU.')
        parser.add_argument('--gradient_accumulation_steps', type=int, help='Amount of gradients to accumulate before performing an optimizer step.')
        parser.add_argument('--learning_rate', type=float, help='The learning rate.')
        parser.add_argument('--lr_scheduler_type', type=str, help='Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis.')
        parser.add_argument('--group_by_length', action='store_true', help='Group sequences into batches with same length.')
        parser.add_argument('--num_train_epochs', type=int, help='Amount of epochs to train.')
        parser.add_argument('--logging_steps', type=int, help='The frequency of update steps after which to log the loss.')
        parser.add_argument('--save_strategy', type=str, help='When to save checkpoints.')
        parser.add_argument('--evaluation_strategy', type=str, help='When to compute eval loss on eval dataset.')
        parser.add_argument('--optim', type=str, help='The optimizer to be used.')

        # LORA-related arguments
        parser.add_argument('--lora_r', type=int, help="Lora R dimension.")
        parser.add_argument('--lora_alpha', type=int, help="Lora alpha.")
        parser.add_argument('--lora_dropout', type=float, help="Lora dropout.")
        parser.add_argument('--from_checkpoint', action='store_true', help="If resuming from checkpoint.")

        # Inference-related arguments
        parser.add_argument('--max_new_tokens', type=int, help="Maximum sequence length for new tokens during inference.")
        parser.add_argument('--top_k', type=float, help="Top-k sampling parameter.")
        parser.add_argument('--top_p', type=float, help="Top-p sampling parameter.")
        parser.add_argument('--temperature', type=float, help="Temperature for sampling.")
        parser.add_argument('--few_shots', type=int)
        
        return parser

if __name__ == "__main__":
    main()