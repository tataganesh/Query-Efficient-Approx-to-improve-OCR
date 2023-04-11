#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=3000M       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=16:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_patch_logs/%j.out
#SBATCH --array=1-4

EXP_NUM=$((453+${SLURM_ARRAY_TASK_ID}))
echo "Running Experiment $EXP_NUM"

module load StdEnv/2020 tesseract/4.1.0

source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate
# source /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/ocr_calls_new/bin/activate

# Wandb Commands
# wandb enabled
# wandb disabled
# wandb offline
wandb login "$WANDB_API_KEY"

PROJECT_HOME="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR"
cd $PROJECT_HOME || { echo "$PROJECT_HOME doesn't exist"; exit 1; }
DATA_PATH="$SLURM_TMPDIR/data"
DATASET_NAME="patch_dataset" # Name should be changed
if [ ! -d $DATA_PATH ]
then
    echo "$DATASET_NAME Dataset extraction started"
    cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/$DATASET_NAME.zip" $SLURM_TMPDIR/
    # cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/0.15/$DATASET_NAME.zip" $SLURM_TMPDIR/
    cd "$SLURM_TMPDIR" || { echo "$SLURM_TMPDIR doesn't exist"; exit 1; }
    unzip "$DATASET_NAME.zip" >> /dev/null
    mv $DATASET_NAME data
    echo "$DATASET_NAME Dataset extracted"
else
    echo "$DATASET_NAME Dataset exists"
fi

# OCR="Tesseract"
OCR="EasyOCR"

if [ $OCR == "Tesseract" ]
then
    CRNN_MODEL_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_17/crnn_warmup/crnn_model_49"
else
    CRNN_MODEL_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_260/crnn_warmup/crnn_model_32"
fi

cd $PROJECT_HOME || { echo "$PROJECT_HOME doesn't exist"; exit 1; }
BATCH_SIZE=1
EPOCH=50
EXP_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/"
CKPT_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/ckpts"
# PREP_MODEL_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_249/ckpts/Prep_model_44"
CER_JSON_PATH="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/pos_dataset_cers.json"
# CRNN_MODEL_PATH="/home/ganesh/scratch/experiment_442/crnn_warmup/crnn_model_199_79.44"

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" job_array)
eval "$command"
