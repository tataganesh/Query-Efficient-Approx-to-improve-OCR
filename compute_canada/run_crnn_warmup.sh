#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=8:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/crnn_warmup_logs/%j.out

EXP_NUM=454
date +%c
echo "Running Experiment $EXP_NUM"
module load StdEnv/2020 tesseract/4.1.0

# OCR="Tesseract"
OCR="EasyOCR"
#OCR="gvision"

if [ $OCR == "gvision" ]; then
    VENV_PATH="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/ocr_calls_new/bin/activate"
elif [ $OCR == "EasyOCR" ] || [ $OCR == "Tesseract" ]; then
    VENV_PATH="/home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate"
else 
    echo "$OCR is not a valid ocr option. Exiting.."
    exit
fi
source "$VENV_PATH"


PROJECT_HOME="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR"

cd $PROJECT_HOME
DATASET_NAME="textarea_dataset"
DATA_PATH="$SLURM_TMPDIR/DATASET_NAME"
DATASET_PATH="/home/ganesh/projects/def-nilanjan/ganesh/datasets/$DATASET_NAME.zip"
echo "Dataset Path - $DATASET_PATH"
if [ ! -d $DATA_PATH ]
then
	echo "Extraction started for $DATASET_NAME"
	cp $DATASET_PATH $SLURM_TMPDIR/
	cd $SLURM_TMPDIR
	unzip $DATASET_NAME.zip >> /dev/null
	# unzip $DATASET_NAME.zip -d $DATASET_NAME >> /dev/null

	echo "$DATASET_NAME unzipped"
else
	echo "Dataset exists"
fi

cd $PROJECT_HOME
BATCH_SIZE=64
EPOCH=50
CRNN_MODEL_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/crnn_warmup/crnn_model"
# CKPT_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_17/crnn_warmup/crnn_model_49"
# CKPT_PATH="/home/ganesh/scratch/experiment_439/crnn_warmup/crnn_model_83_79.48"
mkdir -p $CRNN_MODEL_PATH
# tensorboard --logdir=$TB_LOGS_PATH --host 0.0.0.0 &
echo "Running training script"
python -u train_crnn.py --batch_size $BATCH_SIZE --epoch $EPOCH --crnn_model_path $CRNN_MODEL_PATH --dataset pos --data_base_path $DATA_PATH --ocr $OCR # --lr 5e-05 # --ckpt_path $CKPT_PATH  --train_subset 100 --val_subset 100
#--minibatch_subset random 
# --ckpt_path $CKPT_PATH --start_epoch 6
date +%c