from PIL import Image
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np
import pytesseract
# from tkinter import *
# import skimage.io

# C:\Users\FarookFazni\anaconda3\Scripts\activate

pytesseract.pytesseract.tesseract_cmd = 'F:\\Program Files\\Tesseract-OCR\\tesseract.exe'

st.write("""
# Student Id details extracter
""")


class stdid():
    def __init__(self,img1):
        # img = cv2.imread(img1)
        # img = cv2.resize(img, (1029, 644))
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
        cropped2 = edged[220:600, 540:980]
        keypoints = cv2.findContours(
            edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        width, height = 300, 400
        pts1 = np.float32([location[0], location[3], location[1], location[2]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix1 = cv2.getPerspectiveTransform(pts1, pts2)
        imgId = cv2.warpPerspective(
            img1, matrix1, (width, height), borderValue=(255, 255, 255))
        # imOut = cv2.cvtColor(imgId, cv2.COLOR_BGR2RGB)
        

        ret, thresh1 = cv2.threshold(
            edged, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

            # Appplying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

            # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)

            # Creating a copy of image
        im2 = img1.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            x, y, w, h = 540, 220, 950, 580

                # Drawing a rectangle on copied image
            rect = cv2.rectangle(
                bfilter, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Cropping the text block for giving input to OCR
            cropped = bfilter[y:y + h, x:x + w]

                # Apply OCR on the cropped image
            text = pytesseract.image_to_string(
                cropped, config='--oem 2 --psm 3')
        
        # plt.imshow(imgId)
        st.write(text)
        st.image(imgId)
    
    
        imOut = cv2.cvtColor(imgId, cv2.COLOR_BGR2RGB)
        cv2.imwrite('savedImage.jpg', imOut)
        
        # st.image(imOut, use_column_width=True,clamp = True)
    
        with open('file.txt', mode='w') as f:
            f.write(text)

file = st.file_uploader('Upload an Student ID image', type=['jpg', 'png', 'jpeg'])
# eval("image")

if file is not None:
    my_img = Image.open(file)
    st.image(my_img)
    # my_img = image
    if st.button("Process"):
        frame = np.asarray(my_img)
        frame = cv2.resize(frame,(1029, 644))
        stdid(frame)
        