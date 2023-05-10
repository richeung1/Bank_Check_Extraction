import streamlit as st
import tempfile
from PIL import Image
from Extraction_Script import Extraction
import os
import OCR_Script as text
import torch
import pandas as pd
import base64


st.title("Bank Check Extraction")

# Create a file uploader with a label
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    torch.cuda.empty_cache()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        # st.write('Saved file to', tmp_file.name)

    image_path = tmp_file.name
    print(image_path)

    # Shows uploaded image in the User Interface
    picture = Image.open(image_path)
    st.image(picture, caption='Uploaded Image.', use_column_width=True)

    picture.save('uploaded_check.jpg')

    # Sending uploaded image to computer vision object detection Extraction_Script.py
    extraction = Extraction()
    extraction.results('uploaded_check.jpg')

    folder_name = 'detected_objects'
    folder_path = os.path.join(os.getcwd(), folder_name)
    print(folder_path)

    Images_for_OCR = []
    file_list = os.listdir(folder_path)
    for filename in file_list:
        if filename.startswith('check') or filename.startswith('date') or filename.startswith('payee') or filename.startswith('amount'):
            Images_for_OCR.append(filename)

    print(Images_for_OCR)

    # Shows Resulting Check Image below the original uploaded image
    for filename in file_list:
        if filename.endswith('final.JPG'):
            image_path = ('detected_objects/' + filename)
            img = Image.open(image_path)
            st.image(img, caption=filename, use_column_width=True)
        

    # Getting text results for the 4 images from OCR_Script's TrOCR-handwritten-large model
    model = text.loadmodel()
    results = []
    for i in Images_for_OCR:
        image_path = ('detected_objects/' + i)
        img = Image.open(image_path)
        st.image(img, caption= i, use_column_width=False)

        ocr = text.OCR(model)
        prediction_text= ocr.process_image(image = img)
        results.append([i, prediction_text])

    print('OCR Results for amount, check #, date, and payee:')
    print(results)



    # Download file button
    df = pd.DataFrame(results, columns = ['Object', 'Text'])
    csv = df.to_csv(index = False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.write('OCR Results for amount, check #, date, and payee.')
    link = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download results as CSV file!</a>'
    st.markdown(link, unsafe_allow_html=True)
    st.write(df)



    st.success("Extraction completed!")
else:
    st.warning("Please upload an image file.")



