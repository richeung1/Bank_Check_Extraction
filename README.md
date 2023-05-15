# Bank Check Extraction

## The goal of this is to make bookkeeping more efficient by automating the data validation and data entry of bank checks.

### User_Interface.py is the main file to run. It creates a UI and is the pipeline to connect 2 models together from Extraction_Script.py and OCR_Script.py.

***User-Interface.py does the following:***
1. User can upload an image of a check
2. Shows the check image in the browser
3. Runs the ***Extraction_Script.py*** on the check image
    - ***Extraction_Script.py*** does the following:
        - Takes an image of a bank check
        - Applies custom-trained Yolov3 computer vision model to the image 
        - Detects 4 objects: check number, date, payee, and amount
        - Applies Non-Maximum Supression to remove duplicate objects in the new image
        - Creates a new image with the bounding boxes of the 4 objects
        - Extracts each object as its own image file
4. Runs the ***OCR_Script.py*** on each of the detected objects: check number, date, payee, and amount
    - ***OCR_Script.py*** does the following:
        - Takes an image
        - Applies custom-trained TrOCR-Large-Handwritten transformer model to the image
        - Prints the text from the image
5. Shows all the detected objects in the browser
6. Prints a dataframe in the browser showing the images' names and the OCR text result
7. Allows user the option to download the dataframe as a CSV


### *Other continuous steps:*
1. Finetune Yolov3 model for Extraction_Script to better detect the 4 objects: check number, date, payee, and amount
2. Finetune Microsoft's TrOCR-Large-Handwritten transformer model to more accurately recognize handwriting in the images
3. Make UI look better