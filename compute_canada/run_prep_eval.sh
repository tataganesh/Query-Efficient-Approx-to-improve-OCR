#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=1  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8000       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=01:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_patch_logs/eval/%j.out


module load StdEnv/2020 tesseract/4.1.0
VENV_PATH="/home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate"
# VENV_PATH="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/ocr_calls_new/bin/activate"
source $VENV_PATH

# Command-line arguments
DATASET_NAME=$1
OCR=$2

DATA_PATH="$SLURM_TMPDIR/$DATASET_NAME"

if [ ! -d $DATA_PATH ]
then
    echo "$DATASET_NAME Dataset extraction started"
    cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/$DATASET_NAME.zip" $SLURM_TMPDIR/
    cd $SLURM_TMPDIR
    unzip $DATASET_NAME.zip >> /dev/null
    echo "$DATASET_NAME Dataset extracted"
else
    echo "$DATASET_NAME Dataset exists"
fi
EXP_ID=249
# cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
cd  /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
# PREP_PATH="/home/ganesh/scratch/experiment_$EXP_ID/ckpts"
PREP_PATH=/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_$EXP_ID/ckpts
PREP_NAME="Prep_model_44"

echo "Evaluating EasyOCR $PREP_NAME on Tesseract"
python -u eval_prep.py --prep_path "$PREP_PATH/$PREP_NAME" --dataset $DATASET_NAME --data_base_path $DATA_PATH --ocr $OCR --show_orig

# echo "Evaluating Tesseract $PREP_NAME on EasyOCR"
# python -u eval_prep.py --prep_path "$PREP_PATH/$PREP_NAME" --dataset $DATASET_NAME --data_base_path $DATA_PATH --ocr EasyOCR
