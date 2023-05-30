#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8000M       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=3:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_patch_logs/%j.out

EXP_NUM=502
echo "Running Experiment $EXP_NUM"

# module load gcc/9.3.0 opencv
module load StdEnv/2020 tesseract/4.1.0

OCR="Tesseract"
# OCR="EasyOCR"
#OCR="gvision"

PROJECT_HOME="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR"
if [ $OCR == "gvision" ]; then
    VENV_PATH="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/ocr_calls_new/bin/activate"
elif [ $OCR == "EasyOCR" ] || [ $OCR == "Tesseract" ]; then
    VENV_PATH="/home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate"
else 
    echo "$OCR is not a valid ocr option. Exiting.."
    exit
fi
source "$VENV_PATH"

# wandb offline
# wandb disabled
wandb login $WANDB_API_KEY

cd $PROJECT_HOME
DATASET_NAME="patch_dataset" # Name should be changed
DATA_PATH="$SLURM_TMPDIR/$DATASET_NAME"
if [ ! -d $DATA_PATH ]
then
    echo "$DATASET_NAME Dataset extraction started"
    cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/$DATASET_NAME.zip" $SLURM_TMPDIR/
    # cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/0.15/$DATASET_NAME.zip" $SLURM_TMPDIR/
    cd $SLURM_TMPDIR
    unzip "$DATASET_NAME.zip" >> /dev/null
    echo "$DATASET_NAME Dataset extracted"
else
    echo "$DATASET_NAME Dataset exists"
fi

cd $PROJECT_HOME
EPOCH=50
EXP_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/"
# CRNN_MODEL_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_431/crnn_warmup/crnn_model_49"
CRNN_MODEL_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_17/crnn_warmup/crnn_model_49"
CKPT_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/ckpts"
# PREP_MODEL_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_249/ckpts/Prep_model_44"
CER_JSON_PATH="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/pos_dataset_cers.json"
mkdir -p $CKPT_BASE_PATH
echo "Running training script"


# CRNN_MODEL_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/ckpts/CRNN_model_44"
# PREP_MODEL_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/ckpts/Prep_model_44_63.36"
# PREP_OPTIMIZER_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/ckpts/optim_prep_latest"
# CRNN_OPTIMIZER_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/ckpts/optim_crnn_latest"

 python3 -u patch_cli.py --epoch $EPOCH --data_base_path $DATA_PATH --crnn_model  $CRNN_MODEL_PATH --exp_base_path $EXP_BASE_PATH --exp_name patch_100_prune10 --exp_id $EXP_NUM --inner_limit 1  --cers_ocr_path $CER_JSON_PATH --ocr $OCR --pruning_artifact cers_pos_topk_10

# python3 -u patch_cli.py --epoch $EPOCH --data_base_path $DATA_PATH --crnn_model  $CRNN_MODEL_PATH --exp_base_path $EXP_BASE_PATH --exp_name patch_8_random_bp498 --exp_id $EXP_NUM  --minibatch_subset random --minibatch_subset_prop 0.87 --inner_limit 1  --cers_ocr_path $CER_JSON_PATH --ocr $OCR --prep_model $PREP_MODEL_PATH --optim_crnn_path $CRNN_OPTIMIZER_PATH --optim_prep_path $PREP_OPTIMIZER_PATH --start_epoch 45 --lr_crnn 0.000015 --lr_prep 0.0005 --ocr $OCR