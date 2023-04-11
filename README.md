# Bank_Check_Extraction

The goal of this is to make bookkeeping more efficient by automating the data validation and data entry of bank checks.


The Extraction_Script.py does the following:
1. Takes an image of a bank check 
2. Detects 4 objects: check number, date, payee, and amount
3. Saves it as a new image
4. Applies Non-Maximum Supression to remove duplicate bounding objects
5. Saves another new image
6. Extracts each object as its own jpeg file from the previous new image (Non-Maximum Suppressed image)


To be created/uploaded:
1. OCR_Script.py that can process each of the individual images resulting from Extraction_Script.py.
2. The information (printed and/or handwritten) will then be digitized and exported as a CSV, JSON, or other file.
3. This exported file can then be used to upload into an internal database or be used to upload the bank check information into the appropriate bookkeeping software.