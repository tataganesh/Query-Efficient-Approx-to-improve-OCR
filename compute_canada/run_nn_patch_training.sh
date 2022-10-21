#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8000M       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=10:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_patch_logs/%j.out

EXP_NUM=254
echo "Running Experiment $EXP_NUM"

module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate
# wandb enabled
# wandb disabled
# wandb offline
wandb login $WANDB_API_KEY
cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
DATA_PATH="$SLURM_TMPDIR/data"
# DATASET_NAME="patch_dataset" # Name should be changed
DATASET_NAME="funsd" 
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
cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
BATCH_SIZE=1
EPOCH=50
EXP_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/"
# CRNN_MODEL_PATH="/home/ganesh/scratch/experiment_8/crnn_warmup/crnn_model_29"
CRNN_MODEL_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_17/crnn_warmup/crnn_model_49"
TB_LOGS_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/tb_logs"
CKPT_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/ckpts"
PREP_MODEL_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/ckpts/Prep_model_29"
CER_JSON_PATH="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/pos_dataset_cers.json"
mkdir -p $TB_LOGS_PATH $CKPT_BASE_PATH
# tensorboard --logdir=$TB_LOGS_PATH --host 0.0.0.0 &
echo "Running training script"
# python -u train_nn_patch.py --epoch $EPOCH  --crnn_model  $CRNN_MODEL_PATH --data_base_path $SLURM_TMPDIR --exp_base_path $EXP_BASE_PATH --exp_name patch_90 --exp_id $EXP_NUM  --minibatch_subset random --minibatch_subset_prop 1  --inner_limit 2 --label_impute --warmup_epochs 1 --prep_model $PREP_MODEL_PATH # 4 --minibatch_subset random --minibatch_subset_prop 0.9 # --train_subset_size 50 --val_subset_size 25 
# python -u train_nn_patch.py --epoch $EPOCH  --crnn_model  $CRNN_MODEL_PATH --data_base_path $SLURM_TMPDIR --exp_base_path $EXP_BASE_PATH --exp_name patch_full_tracking --exp_id $EXP_NUM --inner_limit 2 --inner_limit_skip --warmup_epochs 0 --weight_decay 0 --cers_ocr_path $CER_JSON_PATH
# python -u train_nn_patch.py --epoch $EPOCH --data_base_path $SLURM_TMPDIR --crnn_model  $CRNN_MODEL_PATH --exp_base_path $EXP_BASE_PATH --exp_name patch_rangeCER_4_tracking_imp --exp_id $EXP_NUM --minibatch_subset rangeCER --minibatch_subset_prop 0.95 --inner_limit 1 --inner_limit_skip --cers_ocr_path $CER_JSON_PATH --crnn_imputation # --softlabel_tracking_prob 0.9 --train_subset_size 50 --val_subset_size 25  --inner_limit_skip
# python -u train_nn_patch.py --epoch $EPOCH --data_base_path $SLURM_TMPDIR --crnn_model  $CRNN_MODEL_PATH --exp_base_path $EXP_BASE_PATH --exp_name patch_uniform_0.95_skip_inner_tracking_wd0_r100 --exp_id $EXP_NUM --minibatch_subset uniformCER --minibatch_subset_prop 0.95 --inner_limit 2  --inner_limit_skip --warmup_epochs 0 --cers_ocr_path $CER_JSON_PATH --random_seed 100 --weight_decay 0 # --softlabel_tracking_prob 0.9 --train_subset_size 50 --val_subset_size 25  --inner_limit_skip

python -u train_nn_patch.py --epoch $EPOCH --data_base_path $SLURM_TMPDIR --crnn_model  $CRNN_MODEL_PATH --exp_base_path $EXP_BASE_PATH --exp_name funsd_full --exp_id $EXP_NUM --inner_limit 4 --cers_ocr_path $CER_JSON_PATH --dataset_type funsd  


