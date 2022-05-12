"""
The script reads a json file containing CERs details for each receipt image. 
THe format of the json file is

{
    "file_name": {
        "index_label_foldername": cervalue,
        ....
    }
}
E.g.

{
	"receipt_00040_0.png": {
		"0_1_receipt_00040.png": 2.0,
		"1_Home_receipt_00040.png": 1.0,
		"2_Made_receipt_00040.png": 1.0
	}
}

Plots
* Histogram based on AVERAGE CER of receipt images.
* Percentage of easy and hard samples based on different CER thresholds
* Tesseract Accuracy of easy and hard samples based on different CER thresholds

Dataset Split

* Split dataset based on easy and hard samples obtained through different CER thresholds. 
 
    
"""



import json
import matplotlib.pyplot as plt
import shutil
from pprint import pprint
import os

DEST_FOLDER_PATH = '/Users/ganesh/UofA/thesis/datasets/cer_partitioned_datasets'
VAL_FOLDER_PATH = '/Users/ganesh/UofA/thesis/datasets/approx-ocr-grad/patch_dataset/patch_dataset_dev'
TEST_FOLDER_PATH = '/Users/ganesh/UofA/thesis/datasets/approx-ocr-grad/patch_dataset/patch_dataset_test'
CORD_PATH = "/Users/ganesh/UofA/thesis/datasets/approx-ocr-grad/patch_dataset/patch_dataset_train/CORD"
FINDIT_PATH = "/Users/ganesh/UofA/thesis/datasets/approx-ocr-grad/patch_dataset/patch_dataset_train/findit"
RRC_PATH  = "/Users/ganesh/UofA/thesis/datasets/approx-ocr-grad/patch_dataset/patch_dataset_train/RRC"
TRAIN_FOLDER_NAME = 'patch_dataset_train'

SHOW_PLOTS = True
CREATE_DATA_SPLITS = False

cers = json.load(open('all_cers_with_img_data.json', 'r'))
cer_avgs = list()
avgs = list()
labels = list()
total_labels = 0
cer_acc_mapping = dict()
for file_name, cer_groups in cers.items():
    if len(cer_groups) > 0:
        avg_cer = sum(cer_groups.values())/len(cer_groups) # Add all CER values in a document
        cer_avgs.append([file_name, avg_cer])
        total_labels += len(cer_groups)
        labels.append(len(cer_groups))
        correct_count = sum(1 if cer == 0.0 else 0 for cer in cer_groups.values())
        cer_acc_mapping[file_name] = [correct_count/len(cer_groups), correct_count, len(cer_groups)]

file_names, avgs = zip(*cer_avgs)
cer_avgs_sorted = sorted(cer_avgs, key=lambda k:k[1], reverse=True)
if SHOW_PLOTS:
    plt.hist(avgs, bins=20)
    plt.xlabel("Average CER")
    plt.ylabel("Count")
    plt.title("CER Histogram")
    plt.show()
print(len(cer_avgs))

# Show number of easy and hard samples based on CER threshold.
# Show accuracy of split based on CER threshold
hard_samples = list()
easy_samples = list()
hard_accuracies = list()
easy_accuracies = list()
cer_thresholds = [0.05 * i for i in range(0, 10)]
dataset_split_info = dict()
for threshold in cer_thresholds:
    hard_sample_count = 0
    easy_sample_count = 0
    hard_count = 0
    easy_count = 0
    easy_accuracy = 0
    hard_accuracy = 0
    threshold_str = f"{threshold:.2f}"
    dataset_split_info[threshold_str] = dict()
    dataset_split_info[threshold_str]["easy"] = list()
    dataset_split_info[threshold_str]["hard"] = list()
    for i, (file_name, cer) in enumerate(cer_avgs):
        if cer <= threshold:
            easy_sample_count += 1
            easy_accuracy += cer_acc_mapping[file_name][0]
            dataset_split_info[threshold_str]["easy"].append(file_name)
        else:
            hard_sample_count += 1
            hard_accuracy += cer_acc_mapping[file_name][0]
            dataset_split_info[threshold_str]["hard"].append(file_name)
    hard_samples.append(hard_sample_count * 100 / (hard_sample_count + easy_sample_count))
    easy_samples.append(easy_sample_count * 100 / (hard_sample_count + easy_sample_count))
    easy_accuracies.append(easy_accuracy * 100/easy_sample_count)
    hard_accuracies.append(hard_accuracy * 100/hard_sample_count)
    
if SHOW_PLOTS:
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(cer_thresholds, hard_samples, label='hard', marker=".", markersize=10)
    axs[0].plot(cer_thresholds, easy_samples, label='easy', marker=".", markersize=10)
    axs[0].set_xticks(cer_thresholds)
    axs[0].set_yticks(range(0, 101, 10))
    axs[0].set_xlabel("CER threshold")
    axs[0].set_ylabel("% of data")
    axs[0].legend()
    axs[0].set_title("Dataset %")
    # plt.show() 
    axs[1].plot(cer_thresholds, hard_accuracies, label='hard', marker=".", markersize=10)
    axs[1].plot(cer_thresholds, easy_accuracies, label='easy', marker=".", markersize=10)
    axs[1].set_xticks(cer_thresholds)
    axs[1].set_xlabel("CER threshold")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_yticks(range(0, 101, 10))
    axs[1].legend()
    axs[1].set_title("Accuracy for different splits")
    plt.show()

if CREATE_DATA_SPLITS:
    # Create new datasets based on CER threshold
    cer_thresholds = [0.05, 0.1, 0.15, 0.2]
    
    for threshold in cer_thresholds:
        threshold_str = f"{threshold:.2f}"
        easy_samples = dataset_split_info[threshold_str]["easy"]
        hard_samples = dataset_split_info[threshold_str]["hard"]
        DEST_THRESHOLD_FOLDER_PATH = os.path.join(DEST_FOLDER_PATH, threshold_str, "easy", TRAIN_FOLDER_NAME)
        for file_name_full in easy_samples:
            folder_name, file_name =  file_name_full.rsplit("_", 1)
            json_name = file_name.split(".")[0]
            if "receipt" in file_name_full:
                file_path = os.path.join(CORD_PATH, folder_name, file_name)
                json_path = os.path.join(CORD_PATH, folder_name, f"{json_name}.json")
                dest_path = os.path.join(DEST_THRESHOLD_FOLDER_PATH, "CORD", folder_name)
            elif file_name_full.startswith("X"):
                file_path = os.path.join(RRC_PATH, folder_name, file_name)
                json_path = os.path.join(RRC_PATH, folder_name, f"{json_name}.json")
                dest_path = os.path.join(DEST_THRESHOLD_FOLDER_PATH, "RRC", folder_name)
            else:
                file_path = os.path.join(FINDIT_PATH, folder_name, file_name)
                json_path = os.path.join(FINDIT_PATH, folder_name, f"{json_name}.json")
                dest_path = os.path.join(DEST_THRESHOLD_FOLDER_PATH, "findit", folder_name)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            shutil.copy(file_path, dest_path)
            shutil.copy(json_path, dest_path)

        DEST_THRESHOLD_FOLDER_PATH = os.path.join(DEST_FOLDER_PATH, threshold_str, "hard", TRAIN_FOLDER_NAME)
        for file_name_full in hard_samples:
            folder_name, file_name =  file_name_full.rsplit("_", 1)
            json_name = file_name.split(".")[0]
            if "receipt" in file_name_full:
                file_path = os.path.join(CORD_PATH, folder_name, file_name)
                json_path = os.path.join(CORD_PATH, folder_name, f"{json_name}.json")
                dest_path = os.path.join(DEST_THRESHOLD_FOLDER_PATH, "CORD", folder_name)
            elif file_name_full.startswith("X"):
                file_path = os.path.join(RRC_PATH, folder_name, file_name)
                json_path = os.path.join(RRC_PATH, folder_name, f"{json_name}.json")
                dest_path = os.path.join(DEST_THRESHOLD_FOLDER_PATH, "RRC", folder_name)
            else:
                file_path = os.path.join(FINDIT_PATH, folder_name, file_name)
                json_path = os.path.join(FINDIT_PATH, folder_name, f"{json_name}.json")
                dest_path = os.path.join(DEST_THRESHOLD_FOLDER_PATH, "findit", folder_name)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            shutil.copy(file_path, dest_path)
            shutil.copy(json_path, dest_path)