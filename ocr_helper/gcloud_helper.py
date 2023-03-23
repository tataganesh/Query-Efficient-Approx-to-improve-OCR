from torchvision.transforms import ToPILImage, PILToTensor
import sys
sys.path.insert(0, "../")
import properties as properties
import utils
from PIL import Image
from io import BytesIO
import json


import subprocess
from google.cloud import vision


import torch
from torchvision import io

import base64
import time

import traceback

VISION_API_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read())


class GcloudHelper:
    def __init__(
        self, empty_char=properties.empty_char, is_eval=False, mock_response=False
    ):
        """Google Cloud Vision API text revognition class

        Args:
            empty_char (str, optional): Empty character mapping. Defaults to properties.empty_char.
            is_eval (bool, optional): Specify if ocr evaluation is being performed. Defaults to False.
        """
        print("Initializing Gcloud helper")
        self.empty_char = empty_char
        self.is_eval = is_eval
        self.client = vision.ImageAnnotatorClient()
        self.mock_response = mock_response

        self.count_calls = 0
        self.count_exceptions = 0
        print("Gcloud helper initialized")

    def get_image_bytes(self, image):
        img = ToPILImage()(image)
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")  # Store image in memory as a buffer
        return img_buffer.getvalue()

    def get_labels(self, imgs):
        """Obtain text labls for a batch of images

        Args:
            imgs (torch.tensor): Batch of word images

        Returns:
            list[str]: List of labels extracted using the OCR
        """
        labels = []
        for i in range(imgs.shape[0]):
            img_bytes = self.get_image_bytes(imgs[i])
            start = time.time()
            image = vision.Image(content=img_bytes)
            response = self.client.text_detection(image)
            end = time.time()
            try:
                texts = response.text_annotations
                if len(texts) == 0:
                    labels.append(self.empty_char)
                    continue
                # At least one text label was extracted
                label = texts[0].description  # Multiple detections possible. We always pick first detection,
                if label == "":
                    label = self.empty_char
                if self.is_eval:
                    labels.append(label)
                    continue 
                label = utils.get_ununicode(label)
                for c in label:
                    if c not in properties.char_set:
                        label = label.replace(c, "")
                if len(label) > properties.max_char_len:
                    label = self.empty_char
                labels.append(label)
            except:
                print(traceback.format_exc())
                print(f"Image Shape - {imgs[i].shape}")
                print("Response")
                print(response)
                print("Exception Raised. Skipping image")
                print(f"Request-Response time - {(end - start) * 1000}")
                labels.append(self.empty_char)
                self.count_exceptions += 1
                if self.count_exceptions > 20:
                    print(f"More than {self.count_exceptions} exceptions. Exiting...") # Limit number of failures
                    exit()
                continue

        return labels

    def get_labels_fullimage(self, image):
        label_bboxes = list()
        h, w = image.shape[-2:]
        img_bytes = self.get_image_bytes(image)
        print("Sending request")
        start = time.time()
        if self.mock_response:
            response = json.load(
                open("/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/output.json"))
        else:
            response = self.get_response(img_bytes)
        self.count_calls += 1
        print("Response Received")
        end = time.time()
        print(f"Time taken - {(end - start) * 1000}")
        all_text_responses = response["responses"][0]["textAnnotations"]
        # Get all words + bboxes for full document image
        for text_info in all_text_responses:
            bbox = dict()
            bbox["label"] = text_info["description"]
            bbox_info = text_info["boundingPoly"]["vertices"]
            bbox["x1"], bbox["y1"] = bbox_info[0].get("x", 0), bbox_info[0].get("y", 0)
            bbox["x2"], bbox["y2"] = bbox_info[1].get("x", w - 1), bbox_info[1].get("y", 0)
            bbox["x3"], bbox["y3"] = bbox_info[2].get("x", w - 1), bbox_info[2].get("y", h - 1)
            bbox["x4"], bbox["y4"] = bbox_info[3].get("x", 0), bbox_info[3].get("y", h - 1)
            label_bboxes.append(bbox)
        return label_bboxes


if __name__ == "__main__":
    img = Image.open('/home/ganesh/projects/def-nilanjan/ganesh/datasets/1.png').convert("L")
    # img = Image.open(
    #     "/home/ganesh/projects/def-nilanjan/ganesh/datasets/5_Tel_141.png"
    # ).convert("L")
    # file_name = os.path.abspath()
    # img = io.read_image(file_name, io.ImageReadMode.GRAY)
    img = PILToTensor()(img)
    # img = torch.ones((1, 1, 1), dtype=torch.uint8)
    imgs = torch.cat([img])

    obj = GcloudHelper()
    labels = obj.get_labels(imgs)
    print(labels)
