#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8000       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=10:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_area_logs/%j.out

EXP_NUM=135
echo "Running Experiment $EXP_ID"

module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate

wandb disabled
wandb offline
#wandb login $WANDB_API_KEY

cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
DATA_PATH="$SLURM_TMPDIR/data"
if [ ! -d $DATA_PATH ]
then
    echo "VGG Dataset extraction started"
    cp /home/ganesh/projects/def-nilanjan/ganesh/datasets/vgg.zip $SLURM_TMPDIR/
    cd $SLURM_TMPDIR
    unzip vgg.zip >> /dev/null
    mv vgg data
    echo "VGG Dataset extracted"
else
    echo "VGG Dataset exists"
fi
cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
BATCH_SIZE=64
EPOCH=50
EXP_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/"
CRNN_MODEL_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_8/crnn_warmup/crnn_model_29"
TB_LOGS_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/tb_logs"
CKPT_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/ckpts"
CER_JSON_PATH="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/vgg_cers.json"
mkdir -p $TB_LOGS_PATH $CKPT_BASE_PATH
# tensorboard --logdir=$TB_LOGS_PATH --host 0.0.0.0 &
echo "Running training script"
python -u train_nn_area.py --batch_size $BATCH_SIZE --epoch $EPOCH --exp_id $EXP_NUM --exp_base_path $EXP_BASE_PATH --crnn_model  $CRNN_MODEL_PATH --data_base_path $SLURM_TMPDIR --exp_name vgg_random_0.90_tracking --inner_limit_skip --cers_ocr_path $CER_JSON_PATH --minibatch_subset_prop 0.90 --minibatch_subset random
# --minibatch_subset_prop 0.5 --label_impute
