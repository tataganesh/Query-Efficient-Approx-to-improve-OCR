#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=00:20:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/crnn_warmup_logs/eval/%j.out


module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate



date +%c
# DATASET_NAME="textarea_dataset"
# DATASET_NAME="patch_dataset"
# DATASET_NAME="gcp_all_textareas"
DATASET_NAME="gcp_textareas"
DATA_PATH="$SLURM_TMPDIR/$DATASET_NAME"

if [ ! -d $DATA_PATH ]
then
    echo "$DATASET_NAME extraction started"
    # cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/$DATASET_NAME.zip" $SLURM_TMPDIR/
    cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/gcloud_vision/exp249_prep/$DATASET_NAME.zip" $SLURM_TMPDIR/
    cd $SLURM_TMPDIR
    unzip $DATASET_NAME.zip >> /dev/null
    echo "$DATASET_NAME Dataset extracted"
else
    echo "$DATASET_NAME Dataset exists"
fi
# cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
cd  /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
# CRNN_MODEL_PATH="/home/ganesh/scratch/experiment_442/crnn_warmup/"
CRNN_MODEL_PATH="/home/ganesh/scratch/experiment_443/ckpts/"
for i in CRNN_model_21 
do
    echo "Evaluating CRNN ckpt $i"
    # python -u -Wignore eval_crnn.py --crnn_path /home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_435/crnn_warmup/ --dataset pos_textarea --crnn_model_name "crnn_model_$i" --data_base_path $SLURM_TMPDIR
    python -u -Wignore eval_crnn.py --crnn_path $CRNN_MODEL_PATH --dataset pos_textarea --crnn_model_name $i --data_base_path $DATA_PATH
    # python -u -Wignore eval_crnn.py --crnn_path $CRNN_MODEL_PATH --dataset pos --crnn_model_name $i --data_base_path $SLURM_TMPDIR --show_orig --ocr gvision
    echo "\n\n"
done
date +%c

