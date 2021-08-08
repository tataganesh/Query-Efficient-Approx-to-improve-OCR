module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate

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
# cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
cd  /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
python -u eval_prep.py --prep_path ~/scratch/experiment_9/ckpts/ --dataset vgg --prep_model_name Prep_model_26 --data_base_path $SLURM_TMPDIR