# Bank_Check_Extraction

The goal of this is to make bookkeeping more efficient by automating the data validation and data entry of bank checks.


The Extraction_Script.py does the following:
1. Takes an image of a bank check 
2. Detects 4 objects: check number, date, payee, and amount
3. Creates a new image showing the detected objects
4. Applies Non-Maximum Supression to remove duplicate objects in the new image
5. Creates another new image with the duplicates removed
6. Extracts each object as its own JPEG file from the latest new image (duplicates removed)


To be created/uploaded:
1. OCR_Script.py that can apply OCR to each of the individual images that were created from Extraction_Script.py
2. The information (printed and/or handwritten) will then be digitized and exported as a CSV, JSON, or other appropriate file
3. This exported file can then be used to upload into an internal database or be used to upload the bank check information into the appropriate bookkeeping software