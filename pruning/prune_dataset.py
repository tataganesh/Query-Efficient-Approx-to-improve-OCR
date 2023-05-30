import argparse
import json
import sys

sys.path.append("../")  # Hack
from collections import defaultdict
from pprint import pprint
import properties
import os
import methods
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import wandb

prune_method_mapping = {"topk": methods.topk, "FL": methods.facility_location}


def save_hist(data, file_name):
    plt.hist(data, bins=20)
    plt.xlabel("Average CER")
    plt.ylabel("Count")
    plt.title("CER Histogram")
    plt.savefig(f"{file_name}.png")
    plt.close()


class DatasetPruner:
    def __init__(self, args):
        print("Dataset Pruning Arguments")
        pprint(vars(args))
        self.cers_tess_path = args.cers_tess_path
        with open(self.cers_tess_path, "r") as f:
            self.cers = json.load(f)
        self.dataset = args.dataset
        self.method_name = args.prune_method
        self.method = prune_method_mapping[self.method_name]
        self.prune_prop = args.prune_prop
        os.makedirs(
            properties.cer_artifacts_path, exist_ok=True
        )  # In case the artifacts folder  does not exist

    def get_image_metric(self):
        print("Calculating mean CER for each document images...")
        cer_groups = defaultdict(list)
        for strip_name, cer in self.cers.items():
            img_name = strip_name.split("_", 2)[-1]
            cer_groups[img_name].append(cer)
        cer_means = dict()
        for img_name, cers in cer_groups.items():
            cer_means[img_name] = round(sum(cers) / len(cers), 3)
        print("Completed.")
        return cer_means

    def prune(self, cer_means):
        print(
            f"Pruning {self.prune_prop}% of {self.dataset} dataset using {self.method_name} method."
        )
        num_samples = len(cer_means) - int(len(cer_means) * (self.prune_prop / 100))
        pruned_data = self.method(cer_means, num_samples)
        print(
            f"Size before pruning: {len(cer_means)}, Size after pruning: {len(pruned_data)}"
        )
        return pruned_data

    def save_artifact(self, cer_means, file_name):

        file_path = os.path.join(properties.cer_artifacts_path, f"{file_name}.json")
        with open(file_path, "w") as f:
            json.dump(dict(cer_means), f)
        print(f"Saved dataset information at {file_path}")

        if not isinstance(wandb.run.mode, wandb.sdk.lib.disabled.RunDisabled):
            artifact = wandb.Artifact(type="subset_info", name=file_name)
            artifact.add_file(file_path)
            wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains the Prep with VGG dataset")
    parser.add_argument(
        "--prune_method",
        choices=["topk", "FL"],
        default="topk",
        help="Pruning method",
    )
    parser.add_argument(
        "--prune_prop", type=int, help="Proportion of samples to be pruned", default=10
    )
    parser.add_argument(
        "--dataset", choices=["vgg", "pos"], help="Name of dataset to be pruned"
    )
    parser.add_argument(
        "--cers_tess_path",
        help="Path to text strip cers information with respect to Tesseract",
        required=True,
    )

    with open("../wandb_config.json") as fp:
        wandb_config = json.load(fp)
    wandb.init(**wandb_config)

    args = parser.parse_args()
    pruner = DatasetPruner(args)
    cer_means = pruner.get_image_metric()
    cer_file_name = f"cers_{args.dataset}"
    pruner.save_artifact(cer_means, cer_file_name)

    pruned = pruner.prune(cer_means)
    pruned_file_name = f"{cer_file_name}_{args.prune_method}_{args.prune_prop}"
    pruner.save_artifact(pruned, pruned_file_name)
    
    # Visualize histogram before and after pruning
    save_hist(pruned.values(), "new_topk")
    # save_hist(cer_means.values(), "old")
