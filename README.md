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


### *Main.py does the following:*
1. Takes an image of a check
2. Runs the Extraction_Script.py which detects the objects and puts it in a new folder
3. Runs the OCR_Script.py on each of the relevant objects (check number, date, payee, and amount) in the new folder
4. Prints a list showing the object's file name and the OCR text result

### *Other steps:*
1. Create a User Interface for the pipeline
2. Figure out appropriate method of deployment so end-users can benefit from this program
3. Export results as a CSV, JSON, or other appropriate file for uploading into a database or bookkeeping software


### *Other continuous steps:*
1. Finetune Yolov3 model for Extraction_Script to better detect the 4 objects: check number, date, payee, and amount
2. Custom-train Microsoft's TrOCR-Large-Handwritten transformer model on my own manually generated dataset to more accurately capture handwriting in the images