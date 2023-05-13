#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=4000M       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=5:30:00     # DD-HH:MM:SS
#SBATCH --array 1-100%5   # This will launch N jobs, but only allow M to run in parallel
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_patch_logs/sweeps/%j.out

# Each trial in the study will be run in a separate job.
# The Optuna study_name has to be set to be able to continue an existing study.
# d=`date +%d-%m-%y_%H-%M-%S`
# echo $d
OPTUNA_STUDY_NAME="prep_eocr_8_topkcer"
DB_NAME="prep_train_easyocr" 

module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate
DATASET_NAME="patch_dataset"
DATA_PATH="$SLURM_TMPDIR/$DATASET_NAME"

wandb disabled
export WANDB_SILENT=True
if [ ! -d $DATA_PATH ]
then
	echo "Extraction of $DATASET_NAME started"
	cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/$DATASET_NAME.zip" $SLURM_TMPDIR/
	cd $SLURM_TMPDIR
	unzip $DATASET_NAME.zip >> /dev/null
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
python3 -u patch_sweep.py --optuna_db $OPTUNA_DB --optuna_study_name $OPTUNA_STUDY_NAME --data_base_path $DATA_PATH
 
