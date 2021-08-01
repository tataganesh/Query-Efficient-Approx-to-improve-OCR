#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=15:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_area_logs/%j.out

EXP_NUM=9
echo "Running Experiment $EXP_NUM"

module load StdEnv/2020 tesseract/4.1.0
source ocr_bb_calls/bin/activate

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
cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
BATCH_SIZE=64
EPOCH=30
CRNN_MODEL_PATH="/home/ganesh/scratch/experiment_8/crnn_warmup/crnn_model_29"
TB_LOGS_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/tb_logs"
CKPT_BASE_PATH="/content/drive/MyDrive/thesis/black_box_approx_ckpts/experiment_$EXP_NUM/ckpts"
mkdir -p $TB_LOGS_PATH $CKPT_PATH
python init_workspace.py
BATCH_SIZE = 64
EPOCH = 1
# tensorboard --logdir=$TB_LOGS_PATH --host 0.0.0.0 &
echo "Running training script"
#python -u train_crnn.py --batch_size $BATCH_SIZE --epoch $EPOCH --crnn_model_path $CRNN_MODEL_PATH --dataset vgg --tb_logs_path $TB_LOGS_PATH --ocr Tesseract --data_base_path $SLURM_TMPDIR 
# --ckpt_path $CKPT_PATH --start_epoch 29
!python train_nn_area.py --batch_size $BATCH_SIZE --epoch $EPOCH  --ckpt_base_path $CKPT_BASE_PATH --crnn_model  $CRNN_MODEL_PATH --tb_log_path $TB_LOGS_PATH 
#--minibatch_subset random --inner_limit 2