#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=15:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/crnn_warmup_logs/%j.out

module load StdEnv/2020 tesseract/4.1.0
source ocr_bb_calls/bin/activate
cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
cp /home/ganesh/projects/def-nilanjan/ganesh/datasets/vgg.zip $SLURM_TMPDIR/
cd $SLURM_TMPDIR
unzip vgg.zip >> /dev/null
echo "Dataste unzipped"
mv vgg data
cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
BATCH_SIZE=64
EPOCH=30
CRNN_MODEL_PATH='/home/ganesh/scratch/experiment_8/crnn_warmup/crnn_model'
TB_LOGS_PATH='/home/ganesh/scratch/experiment_8/tb_logs'
CKPT_PATH='/home/ganesh/scratch/experiment_8/crnn_warmup/crnn_model_29'
mkdir $TB_LOGS_PATH
python init_workspace.py
# tensorboard --logdir=$TB_LOGS_PATH --host 0.0.0.0 &
echo "Running training script"
python -u train_crnn.py --batch_size $BATCH_SIZE --epoch $EPOCH --crnn_model_path $CRNN_MODEL_PATH --dataset vgg --tb_logs_path $TB_LOGS_PATH --ocr Tesseract --data_base_path $SLURM_TMPDIR 
# --ckpt_path $CKPT_PATH --start_epoch 29