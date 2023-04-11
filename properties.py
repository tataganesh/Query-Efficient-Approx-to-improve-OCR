# POS Text areas
pos_text_dataset_train = "data/textarea_dataset_train"
pos_text_dataset_test = "data/textarea_dataset_test"
pos_text_dataset_dev = "data/textarea_dataset_dev"

# VGG
vgg_text_dataset_train = "data/vgg_train"
vgg_text_dataset_test = "data/vgg_test"
vgg_text_dataset_dev = "data/vgg_dev"

# POS Patches
patch_dataset_train = "data/patch_dataset_train"
patch_dataset_test = "data/patch_dataset_test"
patch_dataset_dev = "data/patch_dataset_dev"


# WildReceipt Patches
wr_dataset_train = "data/wildreceipt_train"
wr_dataset_test = "data/wildreceipt_test"
wr_dataset_dev = "data/wildreceipt_dev"



prep_crnn_ckpts = "ckpts"
crnn_model_path = "./outputs/crnn_trained_model/model"
crnn_tensor_board = "./outputs/crnn_runs/"
prep_model_path = "./outputs/prep_trained_model/"
prep_tensor_board = "prep_runs_tb_logs"
img_out = "img_out"
param_path = "params.txt"
train_subset_size = 50000
val_subset_size = 10000

input_size = (32, 128)
num_workers = 4
char_set = ['`', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '~', '€', '}', '\\', '/']

# tesseract_path = "/usr/share/tesseract-ocr/4.00/tessdata" # Original
tesseract_path = "" # For Google Colab
empty_char = ' '
max_char_len = 100
# max_char_len = 25
