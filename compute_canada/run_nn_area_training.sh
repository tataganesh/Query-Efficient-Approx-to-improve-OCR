#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=12:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_area_logs/%j.out

EXP_NUM=37
echo "Running Experiment $EXP_NUM"

module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate

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
EPOCH=40
EXP_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/"
CRNN_MODEL_PATH="/home/ganesh/scratch/experiment_8/crnn_warmup/crnn_model_29"
TB_LOGS_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/tb_logs"
CKPT_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/ckpts"
mkdir -p $TB_LOGS_PATH $CKPT_BASE_PATH
# tensorboard --logdir=$TB_LOGS_PATH --host 0.0.0.0 &
echo "Running training script"
python -u train_nn_area.py --batch_size $BATCH_SIZE --epoch $EPOCH  --ckpt_base_path $CKPT_BASE_PATH --exp_base_path $EXP_BASE_PATH --crnn_model  $CRNN_MODEL_PATH --data_base_path $SLURM_TMPDIR --exp_name weighted_sampling --warmup_epochs 5
#--minibatch_subset random --inner_limit 2
