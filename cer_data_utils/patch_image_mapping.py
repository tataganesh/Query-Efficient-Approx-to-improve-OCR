"""
The format of  all_cers_textarea.json is as follows

{
    "index_label_foldername": cervalue
}
Using the patch dataset information, we walk through the different dataset folders and 
convert the loaded json to the following format - 

{
	"receipt_00040_0.png": {
		"0_1_receipt_00040.png": 2.0,
		"1_Home_receipt_00040.png": 1.0,
		"2_Made_receipt_00040.png": 1.0
	}
}
    
"""
import os
import json
from pprint import pprint
patch_dataset = '/Users/ganesh/UofA/thesis/datasets/approx-ocr-grad/patch_dataset/patch_dataset_train/'
dataset_names = ['CORD', 'findit', 'RRC']
total = 0
nf = 0
image_data_range = dict()
all_cers = json.load(open('all_cers_textarea.json'))
for dataset in dataset_names:
    folder_path = os.path.join(patch_dataset, dataset)
    for image_folder in os.listdir(folder_path):
        image_folder_path = os.path.join(folder_path, image_folder)
        if not os.path.isdir(image_folder_path):
            continue
        patches_sum = 0 # patches sum are used since the json has been created 
        # using the text area dataset, where all the text strips belonging to an image FOLDER need
        # to be given a unique index. 
        
        for file_ in sorted(os.listdir(image_folder_path), key=lambda k:int(k[0])):
            if ".json" not in file_:
                continue
            img_name = file_[:-4] + "png" # Should be like 1.png, 12.png etc. 
            if not os.path.exists(os.path.join(image_folder_path, img_name)):
                img_name = img_name[:-4] + ".jpg" # If image is not png
            image_data_range[image_folder + "_" + img_name] = dict() #Store all text strips and their CERs belonging to one image
            json_path = os.path.join(image_folder_path, file_)
            with open(json_path, 'r') as f:
                label_json = json.load(f)
            for i in range(0, len(label_json)):
                image_patch_name = f"{patches_sum + i}_{label_json[i]['label']}_{image_folder}.png" # contruct unique text strip ID
                if image_patch_name not in all_cers: # Debug stuff, some text strips are not present since there is a mismatch between patch dataset and text area dataset
                    nf += 1
                else:
                    image_data_range[image_folder + "_" + img_name][image_patch_name] = all_cers[image_patch_name]
                    total += 1
            patches_sum += len(label_json)


# Recreate CER 
all_cers_recreated = dict()
for key, value in image_data_range.items():
    all_cers_recreated.update(value)
    
with open('all_cers_with_img_data.json', 'w') as fp:
    json.dump(image_data_range, fp)