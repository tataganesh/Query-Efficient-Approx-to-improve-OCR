
import optuna
import argparse
import sys
sys.path.append("../")

import train_crnn
import os
from argparse import Namespace
import pathlib

params = {
    "batch_size": 64,
    "epoch": 60,
    "std": 5,
    "dataset": "pos",
    "ocr": None,
    "data_base_path": None,
    "train_subset": None,
    "val_subset": None,
    "dataset": "pos",
    "random_std": True,
    "random_seed": 42,
    "ckpt_path": None,
    "start_epoch": -1,
}

EXP_BASE_PATH = "/home/ganesh/scratch/hyp_sweeps/"


def objective(trial, params, args):
    params["data_base_path"] = args.data_base_path
    trial.set_user_attr("CCID", params["data_base_path"].split(".")[-2])
    crnn_path = pathlib.Path(f"/home/ganesh/scratch/hyp_sweeps/{args.optuna_study_name}/experiment_{trial.number}/crnn_warmup/crnn_model")
    os.makedirs(str(crnn_path.parent), exist_ok=True)
    params["crnn_model_path"] = str(crnn_path)
    lr = trial.suggest_float("lr", 0.00001, 0.0001)
    std = trial.suggest_int("std", 2, 8)
    params["lr"] = lr
    params["std"] = std
    ns = Namespace(**params)
    trainer = train_crnn.TrainCRNN(ns)
    best_val_accuracy, best_val_epoch = trainer.train()
    return best_val_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for sweep for CRNN")

    # args.optuna_db and args.optuna_study_name are command line arguments
    parser.add_argument("--optuna_db", help="Database for storing study")
    parser.add_argument("--optuna_study_name", help="Name of study")
    parser.add_argument("--data_base_path", help="Base path of data")
    args = parser.parse_args()
    print(args)
    
    storage = optuna.storages.RDBStorage(url="sqlite:///" + args.optuna_db, engine_kwargs={"connect_args": {"timeout": 100}})
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=args.optuna_study_name,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, params, args), n_trials=1
    )  # Only execute a single trial at a time, to avoid wasting compute
