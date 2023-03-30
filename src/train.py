DATA_PATH = '/home/kajikawa_r/competition/gradcomp/data'
MODEL_PATH = ''

from make_list import load_data

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
class GradDataset(Dataset):
    def __init__(self,X,y=None)]
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        input = {
            "input_ids": self.X[index]["input_ids"],
            "attention_mask": self.X[index]["attention_mask"],
        }

        if self.y is not None:
            input["labels"] = self.y[index]

        return input
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    f1 = f1_score(p.label_ids, preds)
    return {
        "f1_score": f1,
    }


def train(args, X_train, y_train, X_test):
    y_preds = []
    oof_train = np.zeros((len(X_train),))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    X_train = [tokenizer(text, padding="max_length", max_length=args.max_length, truncation=True) for text in X_train]
    X_test = [tokenizer(text, padding="max_length", max_length=args.max_length, truncation=True) for text in X_test]
    test_dataset = HateSpeechDataset(X_test)

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    for kfold_idx, (tra_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tra = [X_train[i] for i in tra_idx]
        X_val = [X_train[i] for i in val_idx]
        y_tra = [y_train[i] for i in tra_idx]
        y_val = [y_train[i] for i in val_idx]

        training_dataset = HateSpeechDataset(X_tra, y_tra)
        validation_dataset = HateSpeechDataset(X_val, y_val)

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

        training_args = TrainingArguments(
            output_dir=os.path.join(args.save_model_dir, f"fold{kfold_idx}"),
            overwrite_output_dir=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            log_level="critical",
            logging_steps=25,
            save_strategy="steps",
            save_steps=25,
            save_total_limit=3,
            label_smoothing_factor=args.label_smoothing_factor,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model=args.metric_for_best_model,
            seed=args.seed + kfold_idx**2,
            data_seed=args.seed + kfold_idx**2,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience,
                )
            ],
        )
        trainer.train()

        oof_train[val_idx] = np.argmax(trainer.predict(validation_dataset).predictions, axis=1)
        y_pred = trainer.predict(test_dataset).predictions
        y_preds.append(y_pred)

        trainer.save_model(os.path.join(args.save_model_dir, f"fold{kfold_idx}"))

    return oof_train, y_preds

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seed_everything(args.seed)
    
    # set data
    # devがもともとなかった
    x_train = load_data(DATA_PATH,"text.train.txt")
    x_test = load_data(DATA_PATH,"text.test.txt")
    x_dev = load_data(DATA_PATH,"text.dev.txt")

    y_train = load_data(DATA_PATH,"label.train.txt")
    y_dev = load_data(DATA_PATH,"label.dev.txt")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train parameter
    parser.add_argument("--model_name", type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.2)
    parser.add_argument("--metric_for_best_model", type=str, default="f1_score")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    
    # PATH
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--save_model_dir", default="../model")
    parser.add_argument("--sub_file", default="../submission/sub.csv")
    args = parser.parse_args()

    main(args)