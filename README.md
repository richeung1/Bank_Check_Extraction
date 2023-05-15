# Bank Check Extraction

## The goal of this is to make bookkeeping more efficient by automating the data validation and data entry of bank checks.

### This is a minimum viable product. User_Interface.py is the main file to run.

***User-Interface.py does the following:***
1. User can upload an image of a check
2. Shows the check image in the browser
3. Runs the ***Extraction_Script.py*** on the check image
    - ***Extraction_Script.py*** does the following:
        - Takes an image of a bank check
        - Applies custom-trained Yolov3 computer vision model to the image 
        - Detects 4 objects: check number, date, payee, and amount
        - Removes duplicate objects if applicable
        - Creates a new image with the bounding boxes of the 4 objects
        - Extracts each object as its own image file
4. Runs the ***OCR_Script.py*** on the detected objects
    - ***OCR_Script.py*** does the following:
        - Takes an image
        - Applies custom-trained TrOCR-Large-Handwritten transformer model to the image
        - Prints the text from the image
5. Shows all the detected objects in the browser
6. Prints a dataframe in the browser showing the images' names and the OCR text results
7. Allows user the option to download the dataframe as a CSV


### *Other continuous steps:*
1. Finetune Yolov3 model for Extraction_Script to improve detection for the 4 objects: check number, date, payee, and amount
    - Metric used is mAP (Mean Average Precision). It is currently at 89%. Closer to 100% is better.
2. Finetune Microsoft's TrOCR-Large-Handwritten transformer model to more accurately recognize handwriting in the images
    - Metric used is CER (Character Error Rate). It is currenlty at 21%. Closer to 0% is better.
3. Improve User Interface's aesthetic