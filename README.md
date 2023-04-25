# Bank Check Extraction

## The goal of this is to make bookkeeping more efficient by automating the data validation and data entry of bank checks.


### *The Extraction_Script.py does the following:*
1. Takes an image of a bank check
2. Applies custom-trained Yolov3 computer vision model to the image 
3. Detects 4 objects: check number, date, payee, and amount
4. Creates a new image showing the detected objects
5. Applies Non-Maximum Supression to remove duplicate objects in the new image
6. Creates another new image with the duplicates removed
7. Extracts each object as its own JPEG file from the latest new image (duplicates removed)


### *The OCR_Script.py does the following:*
1. Takes an image
2. Applies Microsoft's TrOCR-Large-Handwritten transformer model to the image
3. Shows the image in a separate window
4. Prints the text from the image


### *User-Interface.py does the following:*
1. User can upload an image of a check
2. Shows the check image in the browser
3. Runs the Extraction_Script.py which detects the objects
4. Runs the OCR_Script.py on each of the relevant detected objects: check number, date, payee, and amount
5. Shows all the images in the browser
6. Prints a dataframe in the browser showing the images' names and the OCR text result
7. Allows user the option to download the dataframe as a CSV


### *Other continuous steps:*
1. Finetune Yolov3 model for Extraction_Script to better detect the 4 objects: check number, date, payee, and amount
2. Custom-train Microsoft's TrOCR-Large-Handwritten transformer model on my own manually generated dataset to more accurately capture handwriting in the images
3. Make UI look better