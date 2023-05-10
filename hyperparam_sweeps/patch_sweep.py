import optuna
import argparse
import sys

sys.path.append("../")

import train_nn_patch
import os
from argparse import Namespace
import pathlib
import wandb

wandb.init(project='ocr-calls-reduction', entity='tataganesh')

EXP_BASE_PATH = "/home/ganesh/scratch/hyp_sweeps/"

params = {
    "ocr": "EasyOCR",
    "std": 5,
    "epoch": 25,
    "exp_id": "497",
    "scalar": 1,
    "emb_dim": 256,
    "lr_crnn": 0.0001,
    "lr_prep": 0.00005,
    "exp_name": None,
    "query_dim": 32,
    "crnn_model": "/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_260/crnn_warmup/crnn_model_32",
    "image_prop": None,
    "prep_model": None,
    "random_std": True,
    "inner_limit": 1,
    "random_seed": 42,
    "start_epoch": 0,
    "update_CRNN": False,
    "window_size": 1,
    "decay_factor": 0.7,
    "weight_decay": 0.0005,
    "cers_ocr_path": "/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/pos_dataset_cers.json",
    "exp_base_path": None,
    "warmup_epochs": 0,
    "data_base_path": os.environ.get("SLURM_TMPDIR", "base"),
    "attn_activation": "sigmoid",
    "discount_factor": 1,
    "optim_crnn_path": None,
    "optim_prep_path": None,
    "val_subset_size": None,
    "train_subset_size": None,
    "inner_limit_skip": False,
    "weightgen_method": "decaying",
    "minibatch_subset_prop": 0.87,
    "minibatch_subset": "topKCER"
}


def objective(trial, params, args):
    trial.set_user_attr("CCID", params["data_base_path"].split(".")[-2])
    params["exp_base_path"] = pathlib.Path(f"/home/ganesh/scratch/hyp_sweeps/{args.optuna_study_name}/experiment_{trial.number}/")
    os.makedirs(params["exp_base_path"], exist_ok=True)
    params["exp_id"] = f"study-{args.optuna_study_name}_{trial.number}"
    params['exp_name'] = f"patch_study-{args.optuna_study_name}_{trial.number}"
    params["lr_crnn"] = trial.suggest_float("lr_crnn", 0.00001, 0.001)
    params["lr_prep"] = trial.suggest_float("lr_prep", 0.00001, 0.0005)
    ns = Namespace(**params)
    trainer = train_nn_patch.TrainNNPrep(ns, optuna_trial=trial)
    best_val_accuracy, best_val_epoch = trainer.train()
    trial.set_user_attr("best_val_epoch", best_val_epoch)
    return best_val_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for sweep for CRNN")

    # args.optuna_db and args.optuna_study_name are command line arguments
    parser.add_argument("--optuna_db", help="Database for storing study")
    parser.add_argument("--optuna_study_name", help="Name of study")
    args = parser.parse_args()
    print(args)

    storage = optuna.storages.RDBStorage(
        url="sqlite:///" + args.optuna_db,
        engine_kwargs={"connect_args": {"timeout": 100}}
    )
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=args.optuna_study_name,
        load_if_exists=True,
        pruner=optuna.pruners.ThresholdPruner(lower=35, n_warmup_steps=2)
    )
    study.optimize(
        lambda trial: objective(trial, params, args), n_trials=1
    )  # Only execute a single trial at a time, to avoid wasting compute