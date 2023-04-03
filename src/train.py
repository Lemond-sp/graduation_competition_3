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
    def __init__(self,X,y=None):
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
    f1 = f1_score(p.label_ids, preds, average='micro')
    qwk = cohen_kappa_score(p.label_ids,preds,weights='quadratic')
    return {
        "f1_score": f1,
        'QWK': qwk,
    }

# 訓練と検証
def train(args, X_train, y_train,X_dev,y_dev,X_test):
    y_preds = []

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    X_train = [tokenizer(text, padding="max_length", max_length=args.max_length, truncation=True) for text in X_train]
    X_test = [tokenizer(text, padding="max_length", max_length=args.max_length, truncation=True) for text in X_test]
    X_dev = [tokenizer(text, padding="max_length", max_length=args.max_length, truncation=True) for text in X_dev]
    test_dataset = GradDataset(X_test)
    training_dataset = GradDataset(X_train, y_train)
    validation_dataset = GradDataset(X_dev, y_dev)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name,num_labels=5)
    model.gradient_checkpointing_enable()
    
    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_model_dir, f"{args.model_name_short}"),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        log_level="critical",
        logging_steps=25,
        save_strategy="epoch",
        save_steps=25,
        save_total_limit=3,
        #label_smoothing_factor=args.label_smoothing_factor,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        seed=args.seed,
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
    
    # train
    trainer.train()
    
    # predict
    y_preds = trainer.predict(test_dataset).predictions
    
    # save model
    #trainer.save_model(os.path.join(args.save_model_dir, f"fold{kfold_idx}"))

    return y_preds

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    seed_everything(args.seed)
    
    # set data
    x_train = load_data(args.data_dir,"text.train.txt")
    x_test = load_data(args.data_dir,"text.test.txt")
    x_dev = load_data(args.data_dir,"text.dev.txt")

    y_train = load_data(args.data_dir,"label.train.txt")
    y_dev = load_data(args.data_dir,"label.dev.txt")
    y_train = [i+2 for i in list(map(int,y_train))]
    y_dev = [i+2 for i in list(map(int,y_dev))]

    # train

    y_preds = train(
        args,
        x_train, y_train,x_dev,y_dev,x_test,
    )
    print(y_preds)
    with open('/home/kajikawa_r/competition/gradcomp/ch03/submission/eval.txt','w') as f:
        for line in y_preds:
            f.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train parameter
    parser.add_argument("--model_name", type=str, default="studio-ousia/luke-japanese-large-lite")
    parser.add_argument("--model_name_short", type=str,default="large-lite")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.2)
    parser.add_argument("--metric_for_best_model", type=str, default="QWK") # qwk
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    # PATH
    parser.add_argument("--data_dir", default="/home/kajikawa_r/competition/gradcomp/data")
    parser.add_argument("--save_model_dir", default="/home/kajikawa_r/competition/gradcomp/ch03/model")
    parser.add_argument("--sub_file", default="/home/kajikawa_r/competition/gradcomp/ch03/submission/sub.txt")
    args = parser.parse_args()
    main(args)