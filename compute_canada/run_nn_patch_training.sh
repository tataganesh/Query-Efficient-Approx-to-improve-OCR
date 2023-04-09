#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8000M       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=20:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_patch_logs/%j.out

EXP_NUM=450
echo "Running Experiment $EXP_NUM"

# module load gcc/9.3.0 opencv
module load StdEnv/2020 tesseract/4.1.0


PROJECT_HOME="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR"
VENV_PATH="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/ocr_calls_new/bin/activate"
# VENV_PATH="/home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate"
source "$VENV_PATH"

wandb offline
wandb disabled
# wandb login $WANDB_API_KEY

cd $PROJECT_HOME
DATA_PATH="$SLURM_TMPDIR/data"
DATASET_NAME="patch_dataset" # Name should be changed
if [ ! -d $DATA_PATH ]
then
    echo "$DATASET_NAME Dataset extraction started"
    cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/$DATASET_NAME.zip" $SLURM_TMPDIR/
    # cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/0.15/$DATASET_NAME.zip" $SLURM_TMPDIR/
    cd $SLURM_TMPDIR
    unzip "$DATASET_NAME.zip" >> /dev/null
    mv $DATASET_NAME data
    echo "$DATASET_NAME Dataset extracted"
else
    echo "$DATASET_NAME Dataset exists"
fi

cd $PROJECT_HOME
EPOCH=10
EXP_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/"
# CRNN_MODEL_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_431/crnn_warmup/crnn_model_49"
CRNN_MODEL_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_17/crnn_warmup/crnn_model_49"
# CRNN_MODEL_PATH="/home/ganesh/scratch/experiment_448/crnn_warmup/crnn_model_199_92.71"
# CRNN_MODEL_PATH="/home/ganesh/scratch/experiment_433/ckpts/CRNN_model_2"
CKPT_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/ckpts"
# PREP_MODEL_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_249/ckpts/Prep_model_44"
CER_JSON_PATH="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/pos_dataset_cers.json"
mkdir -p $CKPT_BASE_PATH
echo "Running training script"

# python3 -u train_nn_patch.py --epoch $EPOCH --data_base_path $SLURM_TMPDIR --crnn_model  $CRNN_MODEL_PATH --exp_base_path $EXP_BASE_PATH --exp_name unittest_patch_5 --exp_id $EXP_NUM  --minibatch_subset rangeCER --minibatch_subset_prop 0.99  --inner_limit 1 --cers_ocr_path $CER_JSON_PATH --prep_model $PREP_MODEL_PATH --ocr Tesseract --val_subset_size 25   --train_subset_size 50 
# python3 -u train_nn_patch.py --epoch $EPOCH --data_base_path $SLURM_TMPDIR --crnn_model  $CRNN_MODEL_PATH --exp_base_path $EXP_BASE_PATH --exp_name patch_5_gvision_uniformCER_scratch --exp_id $EXP_NUM  --minibatch_subset rangeCER --minibatch_subset_prop 0.87  --inner_limit 1 --cers_ocr_path $CER_JSON_PATH --val_subset_size 10 --train_subset_size 10 # --inner_limit_skip

# python3 -u train_nn_patch.py --epoch $EPOCH --data_base_path $SLURM_TMPDIR --crnn_model  $CRNN_MODEL_PATH --exp_base_path $EXP_BASE_PATH --exp_name patch_5_uniformCER_noocr_frozencrnn_lr --exp_id $EXP_NUM  --inner_limit 0 --cers_ocr_path $CER_JSON_PATH --prep_model $PREP_MODEL_PATH --lr_prep 0.00001
python3 -u train_nn_patch.py --epoch $EPOCH --data_base_path $SLURM_TMPDIR --crnn_model  $CRNN_MODEL_PATH --exp_base_path $EXP_BASE_PATH --exp_name test_patch_5 --exp_id $EXP_NUM  --inner_limit 1  --minibatch_subset rangeCER --cers_ocr_path $CER_JSON_PATH   --val_subset_size 10   --train_subset_size 10 