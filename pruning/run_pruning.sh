

echo "Running topk Pruning"

python3 prune_dataset.py --dataset vgg --cers_tess_path /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/vgg_cers.json --prune_method topk --prune_prop 10 
python3 prune_dataset.py --dataset vgg --cers_tess_path /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/vgg_cers.json --prune_method topk --prune_prop 20 
python3 prune_dataset.py --dataset vgg --cers_tess_path /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/vgg_cers.json --prune_method topk --prune_prop 30 
python3 prune_dataset.py --dataset vgg --cers_tess_path /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/vgg_cers.json --prune_method topk --prune_prop 40 
python3 prune_dataset.py --dataset vgg --cers_tess_path /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/vgg_cers.json --prune_method topk --prune_prop 50 


echo "Running FL Pruning"
python3 prune_dataset.py --dataset vgg --cers_tess_path /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/vgg_cers.json --prune_method FL --prune_prop 10 
python3 prune_dataset.py --dataset vgg --cers_tess_path /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/vgg_cers.json --prune_method FL --prune_prop 20 
python3 prune_dataset.py --dataset vgg --cers_tess_path /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/vgg_cers.json --prune_method FL --prune_prop 30 
python3 prune_dataset.py --dataset vgg --cers_tess_path /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/vgg_cers.json --prune_method FL --prune_prop 40 
python3 prune_dataset.py --dataset vgg --cers_tess_path /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/vgg_cers.json --prune_method FL --prune_prop 50 