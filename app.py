import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Functions import *
from Sudoku_Solver import *
from Sudoku_Init import *
from PreProcess import *

########################################################################
pathImage = "download.jpeg"
heightImg = 450
widthImg = 450
model = intializePredectionModel()  # LOAD THE CNN MODEL
########################################################################

# Streamlit app configuration
st.set_page_config(layout="wide")
st.title("Sudoku Solver")


with st.sidebar:
    st.header("Upload Sudoku Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("Image uploaded successfully!")
    else:
        st.info("Please upload a Sudoku puzzle image")
col1, col2 = st.columns(2)

# if uploaded_file is not None:
#     # Process the image
#     heightImg = 450
#     widthImg = 450
    
#     with col1:
#         st.header("Original Sudoku")
        
#         # Display original image
#         img = cv2.imread("temp.jpg")
#### 1. PREPARE THE IMAGE
# img = cv2.imread(pathImage)
img = cv2.imread("temp.jpg")
if img is not None:
    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgThreshold = preProcess(img)
    with col1:
        st.header("Orginal Sudoku")
        st.image(img, channels="BGR", caption="Uploaded Sudoku")

    # #### 2. FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

    #### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
    biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
    print(biggest)
    if biggest.size != 0:
        biggest = reorder(biggest)
        print(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        imgDetectedDigits = imgBlank.copy()
        imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

        #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
        imgSolvedDigits = imgBlank.copy()
        boxes = splitBoxes(imgWarpColored)
        print(len(boxes))
        # cv2.imshow("Sample",boxes[65])
        numbers = getPredection(boxes, model)
        print(numbers)
        imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
        numbers = np.asarray(numbers)
        posArray = np.where(numbers > 0, 0, 1)
        print(posArray)


        #### 5. FIND SOLUTION OF THE BOARD
        board = np.array_split(numbers,9)
        print(board)
        try:
            solve(board)
        except:
            pass
        print(board)
        flatList = []
        for sublist in board:
            for item in sublist:
                flatList.append(item)
        solvedNumbers =flatList*posArray
        imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

        # #### 6. OVERLAY SOLUTION
        pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
        imgInvWarpColored = img.copy()
        imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
        inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
        imgDetectedDigits = drawGrid(imgDetectedDigits)
        imgSolvedDigits = drawGrid(imgSolvedDigits)

        imageArray = ([img,imgThreshold,imgContours, imgBigContour],
                    [imgDetectedDigits, imgSolvedDigits,imgInvWarpColored,inv_perspective])
        stackedImage = stackImages(imageArray, 1)
        cv2.imshow('Stacked Images', stackedImage)
        cv2.imshow('Solved', inv_perspective)
        
        with col2:
                st.header("Solved Sudoku")
                st.image(inv_perspective, channels="BGR", caption="Solved Sudoku")
                
                # Add download button for the solved image
                solved_img = Image.fromarray(cv2.cvtColor(inv_perspective, cv2.COLOR_BGR2RGB))
                st.download_button(
                    label="Download Solved Image",
                    data=cv2.imencode('.jpg', inv_perspective)[1].tobytes(),
                    file_name="solved_sudoku.jpg",
                    mime="image/jpeg"
                )
    else:
        st.error("Could not detect a Sudoku grid in the image")
else:
    st.info("Please upload a Sudoku puzzle image to get started")


# from flask import Flask, request, send_file, jsonify
# import cv2
# import numpy as np
# from PIL import Image
# import io
# import os
# from Functions import *
# from Sudoku_Solver import *
# from Sudoku_Init import *
# from PreProcess import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# app = Flask(__name__)

# # Load model once at startup
# heightImg = 450
# widthImg = 450
# model = intializePredectionModel()

# @app.route("/")
# def home():
#     return jsonify({"message": "Sudoku Solver API is running. POST an image to /solve"})

# @app.route("/solve", methods=["POST"])
# def solve_sudoku():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     img = Image.open(file.stream).convert("RGB")
#     img = np.array(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     img = cv2.resize(img, (widthImg, heightImg))

#     imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
#     imgThreshold = preProcess(img)
#     contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     biggest, _ = biggestContour(contours)

#     if biggest.size == 0:
#         return jsonify({"error": "No Sudoku grid detected"}), 400

#     biggest = reorder(biggest)
#     pts1 = np.float32(biggest)
#     pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
#     matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
#     imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

#     boxes = splitBoxes(imgWarpColored)
#     numbers = getPredection(boxes, model)
#     numbers = np.asarray(numbers)
#     posArray = np.where(numbers > 0, 0, 1)
#     board = np.array_split(numbers, 9)

#     try:
#         solve(board)
#     except:
#         return jsonify({"error": "Could not solve Sudoku"}), 500

#     flatList = [item for sublist in board for item in sublist]
#     solvedNumbers = flatList * posArray
#     imgSolvedDigits = displayNumbers(imgBlank.copy(), solvedNumbers)

#     pts2 = np.float32(biggest)
#     pts1 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
#     matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
#     inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)

#     _, buffer = cv2.imencode(".jpg", inv_perspective)
#     return send_file(io.BytesIO(buffer), mimetype="image/jpeg", as_attachment=True, download_name="solved_sudoku.jpg")

# if __name__ == "__main__":
#     app.run()

