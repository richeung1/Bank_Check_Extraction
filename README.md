# Bank_Check_Extraction

The goal of this is to make bookkeeping more efficient by automating the data validation and data entry of bank checks.


The Extraction_Script.py does the following:
1. Takes an image of a bank check
2. Applies custom-trained Yolov3 computer vision model to the image 
3. Detects 4 objects: check number, date, payee, and amount
4. Creates a new image showing the detected objects
5. Applies Non-Maximum Supression to remove duplicate objects in the new image
6. Creates another new image with the duplicates removed
7. Extracts each object as its own JPEG file from the latest new image (duplicates removed)


The OCR_Script.py does the following:
1. Takes an image
2. Applies Microsoft's TrOCR-Large-Handwritten transformer model to the image
3. Shows the image in a separate window
4. Prints the text from the image


Next steps:
1. Create a pipeline to link the Extraction_Script and OCR_Script together
2. Save the results (digitized and/or handwritten text) as a CSV, JSON, or other appropriate file
3. Export this file and use to upload into a database or into relevant bookkeeping software


Other continuous steps:
1. Finetune Yolov3 model for Extraction_Script to better detect the 4 objects: check number, date, payee, and amount
2. Custom-train Microsoft's TrOCR-Large-Handwritten transformer model on my own manually generated dataset to more accurately capture handwriting in the bank check images