import torch
import argparse
import os

import torchvision.transforms as transforms

from datasets import patch_dataset
from datasets import img_dataset
from utils import show_img, compare_labels, get_text_stack, get_ocr_helper
from transform_helper import PadWhite
import properties as properties
from tqdm import tqdm
from pprint import pprint

class EvalPrep():

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.show_txt = args.show_txt
        self.show_img = args.show_img
        self.prep_model_path = args.prep_path
        self.ocr_name = args.ocr
        self.dataset_name = args.dataset
        self.show_orig = args.show_orig

        if self.dataset_name == 'vgg':
            self.test_set = os.path.join(args.data_base_path, properties.vgg_text_dataset_test)
            self.input_size = properties.input_size
        elif self.dataset_name == 'patch_dataset':
            self.test_set = os.path.join(args.data_base_path, properties.patch_dataset_test)
            self.input_size = properties.input_size
        elif self.dataset_name == "wildreceipt":
            self.test_set = os.path.join(args.data_base_path, properties.wr_dataset_test)
            self.input_size = properties.input_size

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.prep_model = torch.load(self.prep_model_path, map_location=self.device).to(self.device)

        self.ocr = get_ocr_helper(self.ocr_name, is_eval=True)

        if self.dataset_name == 'patch_dataset':
            self.dataset = patch_dataset.PatchDataset(self.test_set, pad=True, include_name=True)
        elif self.dataset_name == 'wildreceipt':
            self.dataset = patch_dataset.PatchDataset(self.test_set, pad=True, include_name=True, resize_images=False)
        else:
            transform = transforms.Compose([
                PadWhite(self.input_size),
                transforms.ToTensor(),
            ])
            self.dataset = img_dataset.ImgDataset(
                self.test_set, transform=transform, include_name=True)
        self.loader_eval = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=properties.num_workers)

    def _print_labels(self, labels, pred, ori):
        print()
        print('{:<25}{:<25}{:<25}'.format("GT Label",
                                          "Label for pred", "Label for original"))
        for i in range(len(labels)):
            try:
                print('{:<25}{:<25}{:<25}'.format(labels[i], pred[i], ori[i]))
            except:
                try:
                    print('{:<25}{:<25}{:<25}'.format(
                        labels[i], "*******", ori[i]))
                except:
                    continue

    def eval_area(self):
        print("Eval with ", self.ocr_name)
        self.prep_model.eval()
        pred_correct_count = 0
        ori_correct_count = 0
        ori_cer_final = 0
        pred_cer = 0
        counter = 0

        for images, labels, names in self.loader_eval:
            X_var = images.to(self.device)
            img_preds = self.prep_model(X_var)
            ocr_lbl_pred = self.ocr.get_labels(img_preds.cpu())
            
            if self.show_orig:
                ocr_lbl_ori = self.ocr.get_labels(images.cpu())
                ori_crt_count, ori_cer = compare_labels(ocr_lbl_ori, labels)
                ori_correct_count += ori_crt_count
                ori_lbl_cer += ori_cer
                ori_cer_final = round(ori_lbl_cer/len(labels), 2)
            
            prd_crt_count, prd_cer = compare_labels(
                ocr_lbl_pred, labels)
            pred_correct_count += prd_crt_count
            pred_cer += prd_cer

            if self.show_img:
                show_img(img_preds.detach().cpu(), "Processed images")
            counter += 1
        print()
        print('Correct count from predicted images: {:d}/{:d} ({:.5f})'.format(
            pred_correct_count, len(self.dataset), pred_correct_count/len(self.dataset)))
        if self.show_orig:
            print('Correct count from original images: {:d}/{:d} ({:.5f})'.format(
                ori_correct_count, len(self.dataset), ori_correct_count/len(self.dataset)))
            print('Average CER from original images: {:.5f}'.format(
                ori_cer_final/len(self.dataset)))
        print('Average CER from predicted images: {:.5f}'.format(
            pred_cer/len(self.dataset)))

    def eval_patch(self):
        print("Eval with ", self.ocr_name)
        self.prep_model.eval()
        ori_lbl_crt_count = 0
        ori_lbl_cer = 0
        prd_lbl_crt_count = 0
        prd_lbl_cer = 0
        lbl_count = 0
        counter = 0

        for i, (image, labels_dict, name) in enumerate(self.dataset):
            text_crops, labels = get_text_stack(
                image.detach(), labels_dict, self.input_size)     
            lbl_count += len(labels)
            # print(f"Image - {name}")

            if self.show_orig:
                ocr_labels = self.ocr.get_labels(text_crops)
                if self.dataset_name == 'wildreceipt':
                    ocr_labels = [ocr_lbl.replace(" ", "") for ocr_lbl in ocr_labels]

                ori_crt_count, ori_cer = compare_labels(
                    ocr_labels, labels)
                ori_lbl_crt_count += ori_crt_count
                ori_lbl_cer += ori_cer
                ori_cer = round(ori_cer/len(labels), 2)
                


            image = image.unsqueeze(0)
            X_var = image.to(self.device)
            pred = self.prep_model(X_var)
            pred = pred.detach().cpu()[0]

            pred_crops, labels = get_text_stack(
                pred, labels_dict, self.input_size)
            pred_labels = self.ocr.get_labels(pred_crops)
            if self.dataset_name == 'wildreceipt':
                pred_labels = [pred_lbl.replace(" ", "") for pred_lbl in pred_labels]
            prd_crt_count, prd_cer = compare_labels(
                pred_labels, labels)
            prd_lbl_crt_count += prd_crt_count
            prd_lbl_cer += prd_cer
            prd_cer = round(prd_cer/len(labels), 2)
            # print(f"Original: {ori_crt_count}, Preprocessed: {prd_crt_count}")
            # pprint(list(zip(ocr_labels, pred_labels, labels)))

            if self.show_img:
                show_img(image.cpu())
            if self.show_txt:
                self._print_labels(labels, pred_labels, ocr_labels)
            counter += 1
            if not i % 100:
                print(f"{i} samples completed")

        print()
        print('Correct count from predicted images: {:d}/{:d} ({:.5f})'.format(
            prd_lbl_crt_count, lbl_count, prd_lbl_crt_count/lbl_count))
        if self.show_orig:
            print('Correct count from original images: {:d}/{:d} ({:.5f})'.format(
                ori_lbl_crt_count, lbl_count, ori_lbl_crt_count/lbl_count))
            print('Average CER from original images: ({:.5f})'.format(
                ori_lbl_cer/lbl_count))
        print('Average CER from predicted images: ({:.5f})'.format(
            prd_lbl_cer/lbl_count))
        return prd_lbl_crt_count/lbl_count, prd_lbl_cer/lbl_count

    def eval(self):
        if self.dataset_name in ('patch_dataset', 'wildreceipt'):
            return self.eval_patch()
        else:
            return self.eval_area()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains the CRNN model')
    parser.add_argument('--show_txt', action='store_true',
                        help='prints predictions and groud truth')
    parser.add_argument('--show_img', action='store_true',
                        help='shows each batch of images')
    parser.add_argument('--prep_path', default=properties.prep_model_path,
                        help="specify non-default prep model location")
    parser.add_argument('--dataset', default='patch_dataset',
                        help="performs training with given dataset [patch_dataset, vgg, wildreceipt]")
    parser.add_argument('--ocr', default="Tesseract",
                        help="performs training lebels from given OCR [Tesseract,EasyOCR]")
    parser.add_argument("--batch_size", default=64, type=int,  help='Inference batch size')
    parser.add_argument('--data_base_path',
                        help='Base path training, validation and test data', default=".")
    parser.add_argument('--show_orig', help="Show original flow evaluation", action="store_true")
    
    args = parser.parse_args()
    print(args)
    evaluator = EvalPrep(args)
    evaluator.eval()
