from Extraction_Script import Extraction
import OCR_Script as text
import os
from PIL import Image


# Getting image path and running Extraction_Script's custom-trained Yolov3 model to extract classes/objects into a folder named 'detected_objects'
image_path = 'dummycheck1.JPG'
extraction = Extraction()
extraction.results(image_path)


# Geting access detected_objects folder to get the relevant images
folder_name = 'detected_objects'
folder_path = os.path.join(os.getcwd(), folder_name)
print(folder_path)

Images_for_OCR = []
file_list = os.listdir(folder_path)
for filename in file_list:
    if filename.startswith('check') or filename.startswith('date') or filename.startswith('payee') or filename.startswith('amount'):
        Images_for_OCR.append(filename)

print(Images_for_OCR)
    

# Getting text results for the 4 images from OCR_Script's TrOCR-handwritten-large model
model = text.loadmodel()
results = []
for i in Images_for_OCR:
    image_path = ('detected_objects/' + i)
    img = Image.open(image_path)
    img.show()

    ocr = text.OCR(model)
    prediction_text= ocr.process_image(image = img)
    results.append([i, prediction_text])

print(results)