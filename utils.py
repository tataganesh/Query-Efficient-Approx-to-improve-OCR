import os
import torch
import Levenshtein
import numpy as np
import sys

sys.path.insert(0, "datasets")

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from unidecode import unidecode
import torchvision.utils as utils

import ocr_helper.tess_helper as tess_helper
import ocr_helper.eocr_helper as eocr_helper
import ocr_helper.gcloud_helper as gcloud_helper
from pprint import pprint
import json

def get_char_maps(vocabulary=None):
    if vocabulary is None:
        vocab = (
            ["-"]
            + [chr(ord("a") + i) for i in range(26)]
            + [chr(ord("A") + i) for i in range(26)]
            + [chr(ord("0") + i) for i in range(10)]
        )
    else:
        vocab = vocabulary
    char_to_index = {}
    index_to_char = {}
    cnt = 0
    for c in vocab:
        char_to_index[c] = cnt
        index_to_char[cnt] = c
        cnt += 1
    vocab_size = cnt
    return (char_to_index, index_to_char, vocab_size)


def save_img(images, name, dir, nrow=8):
    img = utils.make_grid(images, nrow=nrow)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, name + ".png"), "PNG")


def show_img(images, title="Figure", nrow=8):
    img = utils.make_grid(images, nrow=nrow)
    npimg = img.numpy()
    plt.figure(num=title)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_ununicode(text):
    text = text.replace("_", "-")
    text = text.replace("`", "'")
    text = text.replace("©", "c")
    text = text.replace("°", "'")
    text = text.replace("£", "E")
    text = text.replace("§", "S")

    index = text.find("€")
    if index >= 0:
        text = text.replace("€", "<eur>")
    un_unicode = unidecode(text)
    if index >= 0:
        un_unicode = un_unicode.replace("<eur>", "€")
    return un_unicode


def pred_to_string(scores, labels, index_to_char, show_text=False):
    preds = []
    # (seq_len, batch, vocab_size) -> (batch, seq_len, vocab_size)
    scores = scores.cpu().permute(1, 0, 2)
    for i in range(scores.shape[0]):
        interim = []
        for symbol in scores[i, :]:
            index = torch.argmax(symbol).item()
            interim.append(index)
        out = ""
        for j in range(len(interim)):
            if len(out) == 0 and interim[j] != 0:
                out += index_to_char[interim[j]]
            elif interim[j] != 0 and interim[j - 1] != interim[j]:
                out += index_to_char[interim[j]]
        preds.append(out)
        if show_text:
            print(labels[i], " -> ", out)
    return preds


def compare_labels(preds, labels):
    correct_count = 0
    total_cer = 0
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
        print(labels)

    lens = len(labels)
    for i in range(lens):
        if preds[i] == labels[i]:
            correct_count += 1
        distance = Levenshtein.distance(labels[i], preds[i])
        total_cer += distance / max(
            1, len(labels[i])
        )  # Handle cases where labels[i] is empty
    return correct_count, total_cer


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def padder(crop, h, w):
    _, c_h, c_w = crop.shape
    pad_left = (w - c_w) // 2
    pad_right = w - pad_left - c_w
    pad_top = (h - c_h) // 2
    pad_bottom = h - pad_top - c_h
    pad = torch.nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 1)
    return pad(crop)


def get_text_stack(image, labels, input_size):
    text_crops = []
    labels_out = []
    for lbl in labels:
        label = lbl["label"]
        x_min = lbl["x_min"]
        y_min = lbl["y_min"]
        x_max = lbl["x_max"]
        y_max = lbl["y_max"]
        text_crop = image[:, y_min:y_max, x_min:x_max]
        text_crop = padder(text_crop, *input_size)
        labels_out.append(label)
        text_crops.append(text_crop)
    return torch.stack(text_crops), labels_out


def get_dir_list(test_dir):
    dir_list = []
    for root, dirs, _ in os.walk(test_dir):
        if not dirs:
            dir_list.append(root)
    return dir_list


def get_file_list(in_dir, filter):
    files = os.listdir(in_dir)
    processed_list = []
    for fil in files:
        if fil[-3:] in filter:
            processed_list.append(os.path.join(in_dir, fil))
    return processed_list


def get_files(in_dir, filter, exclude_files=[]):
    processed_list = []
    for root, _, filenames in os.walk(in_dir):
        for f_name in filenames:
            if f_name in exclude_files:
                continue
            if f_name.endswith(tuple(filter)):
                img_path = os.path.join(root, f_name)
                processed_list.append(img_path)
    return processed_list


def get_noisy_image(image, std=0.05, mean=0):
    noise = torch.normal(mean, std, image.shape)
    out_img = image + noise
    out_img.data.clamp_(0, 1)
    return out_img


def get_ocr_helper(ocr, is_eval=False):
    if ocr == "Tesseract":
        return tess_helper.TessHelper(is_eval=is_eval)
    elif ocr == "EasyOCR":
        return eocr_helper.EocrHelper(is_eval=is_eval)
    elif ocr == "gvision":
        return gcloud_helper.GcloudHelper(is_eval=is_eval)
    else:
        return None


def random_subset(images, labels, num_samples):
    """Get

    Args:
        images (torch.tensor): Input images
        labels (torch.tensor): Input labels
        subset (int): Number of random samples.

    Returns:
        tuple: Return subset of images and labels. Chosen randomly.
    """
    num_images = images.shape[0]
    rand_indices = torch.randperm(num_images)[:num_samples]
    return images[rand_indices], [labels[i] for i in rand_indices], rand_indices


def create_dirs(dirs):
    for x in dirs:
        if not os.path.exists(x):
            os.mkdir(x)


def attention_debug(self, loss_weights, text_crop_names):
    self.attn_outputs = defaultdict(lambda: [])
    self.attn_forward_hook1 = (
        self.attention_model.loss_coef_layer.register_forward_hook(
            self.get_layer_input("loss_w_layer")
        )
    )
    self.attn_forward_hook2 = self.attention_model.Wq.register_forward_hook(
        self.get_layer_input("word_embs")
    )
    print(f"Loss Weights = {loss_weights}")
    print("\nHistory")
    for crop_name in text_crop_names:
        if crop_name in self.tracked_labels:
            print(self.tracked_labels[crop_name])
    print("\nAttention Scores")
    pprint(self.attn_outputs["loss_w_layer"])
    print("\nLinear Layer Weights")
    print(
        self.attention_model.loss_coef_layer.weight,
        self.attention_model.loss_coef_layer.bias,
    )
    print("\n Word Embeddings")
    print(self.attn_outputs["word_embs"])
    self.attn_forward_hook1.remove()
    self.attn_forward_hook2.remove()


def get_layer_input(self, name):
    def hook(model, input, output):
        self.attn_outputs[name].append(input[0].detach())
    return hook


def save_json(metrics, json_path, wandb_obj=None):
    with open(json_path, "w") as f:
        json.dump(metrics, f)
    if wandb_obj is not None:
        wandb_obj.save(json_path)
