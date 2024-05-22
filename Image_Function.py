import cv2
import sys
import imutils
import numpy as np
from statistics import mode
import PySimpleGUI as sg
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image, ImageTk
from myfile.Util import *

def Image_function():
                                        #------------------------------#
                                        # Các hàm và tham số cần thiết #
                                        #------------------------------#
    
    BAR_WIDTH = 20                     # Độ rộng của cột biểu đồ
    BAR_SPACING = 52                   # Khoảng cách giữa các cột
    EDGE_OFFSET = 10                   # Khoảng cách giữa cột đầu tiên và lề
    GRAPH_SIZE = DATA_SIZE = (500,500) # Size của biểu đồ
    
        
    img_path = sg.popup_get_file('Image') # Tải ảnh
    if img_path is None:
        return
    
    # Các tham số để tải model
    detection_model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    emotion_model_path = 'model/XCEPTION.81-0.64.hdf5' # Thay model 

    # Tải model
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    emotion_labels = ["Angry","Disgust","Scared", "Happy", "Sad", "Surprised","Neutral"]
    
                                #---------------------------------------------#
                                # Tạo các layout được hiển thị trong phần mềm #
                                #---------------------------------------------#
                
    colcamera_layout = [[sg.Text("Image",font=("Helvetica",15), size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="Image")]]
    colcamera = sg.Column(colcamera_layout, element_justification='center')

    colprobability_layout = [[sg.Text("Probability",font=("Helvetica",15), size=(60, 1), justification="center")],
                            [sg.Graph(GRAPH_SIZE, (0,-400), DATA_SIZE, key="Graph")]]
    colprobability = sg.Column(colprobability_layout, element_justification='center')

    colslayout = [colcamera, colprobability]

    layout = [[colslayout]]

    window = sg.Window("Facial Expression Recognition", layout,finalize=True, size=(1100,576))

    graph = window["Graph"] 

                                            #---------------------------#
                                            # Dự đoán cảm xúc khuôn mặt #
                                            #---------------------------#

    orig_frame = cv2.imread(img_path)
    frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5)

    if len(faces) != 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = emotion_labels[preds.argmax()]

        draw_bounding_box((fX, fY, fW, fH), orig_frame, (0,0,255))
        draw_text((fX, fY),orig_frame, label, (0,0,255) , 0, -45, 1, 1)

        orig_frame = imresize(orig_frame, (480, 640))


                                                #-------------------#
                                                # Chạy chương trình #
                                                #-------------------#   

    while True:
        event, values = window.read(timeout=20)

        if event == sg.WIN_CLOSED:
            break
            
                                                #-----------------------#
                                                # Hiển thị lên màn hình #
                                                #-----------------------#
    
        imgbytes = cv2.imencode(".png", orig_frame)[1].tobytes()
        window["Image"].update(data=imgbytes)
        
                                                #-----------------------------#
                                                # Vẽ biểu đồ và tính xác suất #
                                                #-----------------------------#  
    
        for (i, (emotion, prob)) in enumerate(zip(emotion_labels, preds)):
            graph_value = int(prob*300)# Tính giá trị của cột
            pro_text = "{:.2f}%".format(prob * 100) # Tính xác suất các biểu cảm
            
            graph.draw_rectangle(top_left=(i * BAR_SPACING + EDGE_OFFSET, graph_value),
                                bottom_right=(i * BAR_SPACING + EDGE_OFFSET + BAR_WIDTH, 0),
                                fill_color='blue') # Vẽ biểu đồ
            graph.draw_text(text=pro_text, location=(i*BAR_SPACING+EDGE_OFFSET+15, graph_value+10), font='_ 8')  # Ghi số liệu trên cột
            graph.draw_text(text=emotion, location=(i*BAR_SPACING+EDGE_OFFSET+10, -30), font='_ 8') # Ghi tên các biểu cảm