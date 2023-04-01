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

def predict(args,X_test):
    y_preds = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    X_test = [tokenizer(text, padding="max_length", max_length=args.max_length, truncation=True) for text in X_test]
    test_dataset = GradDataset(X_test)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name,num_labels=5)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    
    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_model_dir, f"{args.model_name_short}"),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        logging_strategy="epoch", # 学習のロギング（損失値、学習率の状況）
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        # log_level="critical",
        save_strategy="epoch",
        save_total_limit=3, # limit the total amount of checkpoints. Deletes the older checkpoints.
        #label_smoothing_factor=args.label_smoothing_factor,
        remove_unused_columns=False, # https://zenn.dev/ken_11/articles/2e54faf7ac3014 自前のデータセットの場合は、Falseにするのがいい（datasets.Datasetと同じ構造でない限り）
        load_best_model_at_end=True, # load the best model after training
        metric_for_best_model=args.metric_for_best_model, # 
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )
    
    # predict
    y_preds = trainer.predict(test_dataset).predictions
    y_preds = np.argmax(y_preds, axis=1)
    # save model
    #trainer.save_model(os.path.join(args.save_model_dir, f"fold{kfold_idx}"))

    return y_preds

def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    seed_everything(args.seed)
    
    # set data
    x_test = load_data(args.data_dir,"text.test.txt")

    # train

    y_preds = predict(
        args,
        x_test,
    )
    print(y_preds)
    # submit
    with open('/home/kajikawa_r/competition/gradcomp/ch03/submission/eval_test03.txt','w') as f:
        for y_pred in y_preds:
            y_pred = int(y_pred) - 2
            y_pred = str(y_pred)
            f.write(y_pred + "\n")

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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.2)
    parser.add_argument("--metric_for_best_model", type=str, default="QWK") # qwk
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    # PATHmodel_path
    parser.add_argument("--model_path", default="/home/kajikawa_r/competition/gradcomp/ch03/model/large-lite/checkpoint-28125/pytorch_model.bin")
    parser.add_argument("--data_dir", default="/home/kajikawa_r/competition/gradcomp/data")
    parser.add_argument("--save_model_dir", default="/home/kajikawa_r/competition/gradcomp/ch03/model")
    parser.add_argument("--sub_file", default="/home/kajikawa_r/competition/gradcomp/ch03/submission/sub.txt")
    args = parser.parse_args()
    main(args)