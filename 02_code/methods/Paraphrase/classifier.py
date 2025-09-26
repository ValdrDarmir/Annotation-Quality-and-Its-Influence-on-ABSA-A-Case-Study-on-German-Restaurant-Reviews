import pandas as pd
import numpy as np
import torch
import os, sys
import transformers
import re
import json
import time

utils = os.path.abspath('/home/niklasdonhauser/utils') # Relative path to utils scripts
sys.path.append(utils)

from config import Config
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorForSeq2Seq
from evaluation import createResults, convertLabels
from preprocessing import loadDataset, splitForEvalSetting, POLARITY_MAPPING_POL_TO_TERM, POLARITY_MAPPING_TERM_TO_POL, TEXT_TEMPLATE, TEXT_PATTERN, IT_TOKEN, OUTPUT_KEYS

from datetime import datetime, timedelta
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class ABSADataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

class ParaphraseABSA:
    def __init__(self, args):
        self.args = args
        self.task = args.task
        self.model_name_or_path = args.model_name_or_path
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name_or_path)
        self.gpu_count = torch.cuda.device_count()

        self.pol_to_term, self.term_to_pol, self.text_template, self.text_pattern, self.it_token = self.loadPhraseDicts(args.lang)
        
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
        
        print(f"Device count: {self.gpu_count}")
    
    def loadPhraseDicts(self, lang):
        """Loads language-specific dictionaries for category mapping and aspect terms."""
        
        return POLARITY_MAPPING_POL_TO_TERM, POLARITY_MAPPING_TERM_TO_POL, TEXT_TEMPLATE, TEXT_PATTERN, IT_TOKEN
            
    def preprocessData(self, data):
        """Tokenizes text and processes labels into T5 training format."""
        def labelToText(sample):
            aspect_term_text = sample[2] if sample[2] != "NULL" else self.it_token
            return self.text_template.format(
                ac_text=sample[0].replace('-', ' '),
                polarity_text=self.pol_to_term.get(sample[1], sample[1]),
                aspect_term_text=aspect_term_text
            )

        input_texts = data["text"].tolist()
        output_texts = [' [SSEP] '.join(map(labelToText, labels)) for labels in data['labels']]

        print('Train dataset snippet:\n')
        print(input_texts[10])
        print(output_texts[10])
        
        input_encodings = self.tokenizer(input_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        output_encodings = self.tokenizer(output_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        
        return ABSADataset(input_encodings, output_encodings['input_ids'])

    def createModel(self):
        """Initializes and loads the model."""
        print('Creating model of type ', T5ForConditionalGeneration)
        print('Loading model ', self.model_name_or_path)
        return T5ForConditionalGeneration.from_pretrained(self.model_name_or_path)

    def extractLabels(self, paraphrased_text):
        extracted_labels = []
        sents = [s.strip() for s in paraphrased_text.split('[SSEP]')]
        for sent in sents:
                match = re.match(self.text_pattern, sent)
                if match:
                    try:
                        extracted_labels.append([match.group(1).strip(), self.term_to_pol[match.group(2)], 'NULL' if match.group(3) == self.it_token else match.group(3)])
                    except KeyError:
                        print(f"Extraction error for sentence: '{sent}'")
                        print(f"Matched phrases: '{match}'")
        return extracted_labels

    def computeMetrics(self, eval_pred):
        predictions, ground_truth = eval_pred

        predictions_raw = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
        predictions_decoded = self.tokenizer.batch_decode(predictions_raw, skip_special_tokens=True)

        ground_truth = np.where(ground_truth != -100, ground_truth, self.tokenizer.pad_token_id)
        ground_truth_decoded = self.tokenizer.batch_decode(ground_truth, skip_special_tokens=True)

        print('Test dataset prediction:\n')
        print('Gold: ', ground_truth_decoded[10])
        print('Pred: ', predictions_decoded[10])
            
        pred_labels = [self.extractLabels(prediction) for prediction in predictions_decoded]
        gold_labels = [self.extractLabels(ground_truth) for ground_truth in ground_truth_decoded]
        
        pred_labels, _ = convertLabels(pred_labels, self.task, self.label_space)
        gold_labels, _ = convertLabels(gold_labels, self.task, self.label_space)

        results = createResults(pred_labels, gold_labels, self.label_space, self.task)
        
        return {'results': results, 'predictions': pred_labels, 'golds': gold_labels}

    def trainModel(self, train_dataset):

        adjusted_batch = int(self.args.per_device_train_batch_size/(self.gpu_count * self.args.gradient_accumulation_steps))
        
        training_args = Seq2SeqTrainingArguments(
            output_dir = '/home/niklasdonhauser/Paraphrase/outputs',
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=adjusted_batch,
            per_device_eval_batch_size=16,
            save_strategy="no",
            logging_dir="logs",
            logging_steps=100,
            logging_strategy="epoch",
            report_to="none",
            predict_with_generate=True,
            generation_max_length=512 if self.args.dataset == 'software/v3' else 256,
            weight_decay=0.01,
            gradient_accumulation_steps = self.args.gradient_accumulation_steps,
            seed=self.args.seed
        )

        trainer = Seq2SeqTrainer(
            model_init=self.createModel,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.computeMetrics
        )
        
        print("Using the following hyperparameters: lr=" + str(self.args.learning_rate) + " - epochs=" + str(self.args.num_train_epochs) + " - batch=" + str(self.args.per_device_train_batch_size*self.gpu_count))

        start_time = time.time()
        
        trainer.train()

        end_time = time.time()
        training_duration = end_time - start_time

        trainer_args = {}
        trainer_args.update({
            "model_name": self.model_name_or_path,
            "task": self.task,
            "data_setting": self.args.data_setting,
            "dataset": self.args.dataset,
            "per_device_train_batch_size": self.args.per_device_train_batch_size,
            "learning_rate": self.args.learning_rate,
            "num_train_epochs": self.args.num_train_epochs,
            "eval_type": self.args.eval_type,
            "train_runtime": training_duration
        })

        return trainer, trainer_args

    def savePredictions(self, result, preds, golds, args, config, eval_type):
        """Evaluates the model and saves results."""
        output_path = f"{args.output_dir}/{args.task}_{args.dataset.replace('/', '-')}_{args.eval_type}_{args.data_setting}-{eval_type[0]}_{round(args.learning_rate,9)}_{args.per_device_train_batch_size}_{args.num_train_epochs}_{args.seed}/" 
        os.makedirs(output_path, exist_ok=True)
        if self.task == 'acd':
            pd.DataFrame.from_dict(result[0]).transpose().to_csv(f"{output_path}metrics_asp.tsv", sep="\t")
                
        else:
            for idx, name in enumerate(["asp", "asp_pol", "pairs", "pol", "phrases"]):
                pd.DataFrame.from_dict(result[idx]).transpose().to_csv(f"{output_path}metrics_{name}.tsv", sep="\t")

        try:
            matched_samples = [
                {"predictions": pred, "gold_labels": gold}
                for pred, gold in zip(preds, golds)
            ]
            print(matched_samples[:5])
            with open(os.path.join(output_path, 'predictions.json'), "w", encoding="utf-8") as f:
                json.dump({"test": matched_samples}, f, indent=4, ensure_ascii=False)

        except:
            pass

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        
        config.update({
            "gpu_util": used_memory
        })
        
        with open(os.path.join(output_path, 'config.json'), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            
                
    def train_eval(self):

        args = self.args

        train_df, eval_df, self.label_space = splitForEvalSetting(loadDataset(args.data_path, args.dataset, args.data_setting), args.eval_type)

        train = self.preprocessData(train_df)

        eval_orig = self.preprocessData(eval_df)

        trainer, trainer_args = self.trainModel(train)

        if 'dev' in args.eval_type:
            res = trainer.evaluate(eval_dataset=eval_orig)
            results_orig, preds, golds = res['eval_results'], res['eval_predictions'], res['eval_golds']
            self.savePredictions(results_orig, preds, golds, args, trainer_args, 'orig')

        else:
            if args.dataset == 'transport':
                
                _, eval_df, _ = splitForEvalSetting(loadDataset(args.data_path, args.dataset, 'dia'), args.eval_type)
                eval_dia = self.preprocessData(eval_df)

                res = trainer.evaluate(eval_dataset=eval_dia)
                results_dia, preds, golds = res['eval_results'], res['eval_predictions'], res['eval_golds']
                self.savePredictions(results_dia, preds, golds, args, trainer_args, 'dia')

            res = trainer.evaluate(eval_dataset=eval_orig)
            results_orig, preds, golds = res['eval_results'], res['eval_predictions'], res['eval_golds']
            self.savePredictions(results_orig, preds, golds, args, trainer_args, 'orig')
            
if __name__ == "__main__":

    config = Config()
    set_seed(config.seed)
        
    absa = ParaphraseABSA(config)
    absa.train_eval()

