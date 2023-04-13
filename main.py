# Installing dependencies
from imageai.Detection.Custom import CustomObjectDetection
from PIL import Image
import cv2
import os
import torch
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
import Extraction_Script as Predict

# Getting image path
image_path = 'dummycheck1.JPG'
Predict.Extraction.results(image_path)


