module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate


DATA_PATH="$SLURM_TMPDIR/data"
DATASET_NAME="vgg"
if [ ! -d $DATA_PATH ]
then
    echo "$DATASET_NAME extraction started"
    cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/$DATASET_NAME.zip" $SLURM_TMPDIR/
    cd $SLURM_TMPDIR
    unzip $DATASET_NAME.zip >> /dev/null
    mv $DATASET_NAME data
    echo "$DATASET_NAME Dataset extracted"
else
    echo "$DATASET_NAME Dataset exists"
fi
# cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
cd  /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
for i in 49
do
    echo "Evaluating CRNN ckpt $i"
    python -u -Wignore eval_crnn.py --crnn_path /home/ganesh/scratch/experiment_262/crnn_warmup --dataset vgg --crnn_model_name "crnn_model_$i" --data_base_path $SLURM_TMPDIR --ocr EasyOCR
    echo "\n\n"
done
