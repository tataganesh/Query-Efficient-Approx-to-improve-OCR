import argparse
import properties
import wandb
import datetime
from train_nn_area import TrainNNPrep
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains the Prep with VGG dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="input batch size")
    parser.add_argument(
        "--lr_crnn",
        type=float,
        default=0.0001,
        help="CRNN learning rate, not used by adadealta",
    )
    parser.add_argument(
        "--scalar",
        type=float,
        default=1,
        help="scalar in which the secondary loss is multiplied",
    )
    parser.add_argument(
        "--lr_prep",
        type=float,
        default=0.00005,
        help="prep model learning rate, not used by adadealta",
    )
    parser.add_argument("--epoch", type=int, default=50, help="number of epochs")
    parser.add_argument(
        "--warmup_epochs", type=int, default=0, help="number of warmup epochs"
    )
    parser.add_argument(
        "--std",
        type=int,
        default=5,
        help="standard deviation of Gussian noice added to images (this value devided by 100)",
    )
    parser.add_argument(
        "--inner_limit",
        type=int,
        default=2,
        help="number of inner loop iterations in Alogorithm 1. Minimum value is 1.",
    )
    parser.add_argument(
        "--inner_limit_skip",
        help="In the first inner limit loop, do NOT add noise to the image. Added to ease label imputation",
        action="store_true",
    )
    parser.add_argument(
        "--crnn_model",
        help="specify non-default CRNN model location. By default, a new CRNN model will be used",
    )
    parser.add_argument(
        "--prep_model",
        help="specify non-default Prep model location. By default, a new Prep model will be used",
    )
    parser.add_argument(
        "--data_base_path",
        help="Base path training, validation and test data",
        default=".",
    )
    parser.add_argument(
        "--ocr",
        default="Tesseract",
        help="performs training labels from given OCR [Tesseract,EasyOCR]",
    )
    parser.add_argument(
        "--random_std",
        action="store_false",
        help="randomly selected integers from 0 upto given std value (devided by 100) will be used",
        default=True,
    )
    parser.add_argument(
        "--exp_base_path",
        default=".",
        help="Base path for experiment. Defaults to current directory",
    )
    parser.add_argument(
        "--minibatch_subset", help="Specify method to pick subset from minibatch."
    )
    parser.add_argument(
        "--minibatch_subset_prop",
        default=0.5,
        type=float,
        help="If --minibatch_subset is provided, specify percentage of samples per mini-batch.",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Starting epoch. If loading from a ckpt, pass the ckpt epoch here.",
    )
    parser.add_argument(
        "--train_subset_size", help="Subset of training size to use", type=int
    )
    parser.add_argument("--val_subset_size", help="Subset of val size to use", type=int)
    parser.add_argument("--lr_scheduler", help="Specify scheduler to be used")
    parser.add_argument(
        "--exp_name",
        default="default_exp",
        help="Specify name of experiment (JVP Jitter, Sample Dropping Etc.)",
    )
    parser.add_argument("--exp_id", help="Specify unique experiment ID")
    parser.add_argument("--cers_ocr_path", help="Cer information json")
    parser.add_argument(
        "--weightgen_method",
        help="Method for generating loss weights for tracking",
        default="decaying",
        choices=["levenshtein", "self_attention", "decaying"],
    )
    parser.add_argument(
        "--window_size", help="Window Size if tracking is enabled", type=int, default=1
    )
    parser.add_argument(
        "--decay_factor",
        help="Decay factor for decaying loss weight generation",
        type=float,
        default=0.7,
    )

    args = parser.parse_args()
    print(vars(args))
    wandb.init(project="ocr-calls-reduction", entity="tataganesh", tags=["VGG"])
    wandb.config.update(vars(args))
    wandb.run.name = f"{args.exp_name}"

    trainer = TrainNNPrep(args)

    start = datetime.datetime.now()
    trainer.train()
    end = datetime.datetime.now()

    with open(os.path.join(args.exp_base_path, properties.param_path), "w") as f:
        f.write(f"{str(start)}\n")
        f.write(f"{str(args)}\n")
        f.write(f"{str(end)}\n")