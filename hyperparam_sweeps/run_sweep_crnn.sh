#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=4000M       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=5:30:00     # DD-HH:MM:SS
#SBATCH --array 1-60%8   # This will launch N jobs, but only allow M to run in parallel
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/crnn_warmup_logs/sweeps/%j.out

# Each trial in the study will be run in a separate job.
# The Optuna study_name has to be set to be able to continue an existing study.
# d=`date +%d-%m-%y_%H-%M-%S`
# echo $d
OPTUNA_STUDY_NAME=crnn_lr_std
DB_NAME="crnn_train" 

module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate
DATASET_NAME="gcp_textareas"
DATA_PATH="$SLURM_TMPDIR/$DATASET_NAME"
wandb disable
if [ ! -d $DATA_PATH ]
then
	echo "Extraction started"
	cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/gcloud_vision/exp249_prep/$DATASET_NAME.zip" $SLURM_TMPDIR/
	cd $SLURM_TMPDIR
	unzip $DATASET_NAME.zip >> /dev/null
	# unzip $DATASET_NAME.zip -d $DATASET_NAME >> /dev/null

	echo "$DATASET_NAME unzipped"
else
	echo "$DATASET_NAME exists"
fi

cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/hyperparam_sweeps

echo "Running Hyperparameter study $OPTUNA_STUDY_NAME. Saving in DB $DB_NAME"

# More details regarding study - Tune Learning Rate 

# Specify a path in your home, or on project.
OPTUNA_DB=/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/hyperparam_sweeps/optuna_studies/$DB_NAME.db

# Launch your script, giving it as arguments the database file and the study name
python3 -u crnn_sweep.py --optuna_db $OPTUNA_DB --optuna_study_name $OPTUNA_STUDY_NAME --data_base_path $DATA_PATH
 
