import time
import cv2
import PySimpleGUI as sg
from keras.models import load_model
import numpy as np
from statistics import mode
from myfile.Util import *

def Camera_function():
                                        #------------------------------#
                                        # Các hàm và tham số cần thiết #
                                        #------------------------------#
    
    BAR_WIDTH = 20                     # Độ rộng của cột biểu đồ
    BAR_SPACING = 52                   # Khoảng cách giữa các cột
    EDGE_OFFSET = 10                   # Khoảng cách giữa cột đầu tiên và lề
    GRAPH_SIZE = DATA_SIZE = (500,500) # Size của biểu đồ
    
    # Các tham số để tải model
    detection_model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    emotion_model_path = 'model/XCEPTION.81-0.64.hdf5' # Thay model
    emotion_labels = ["Angry","Disgust","Scared", "Happy", "Sad", "Surprised","Neutral"]

    # Các tham số cho bounding box
    frame_window = 10
    emotion_offsets = (20, 40)

    # Tải model
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # Lấy thông số đầu vào của model để phân loại
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # Hàm tính toán
    emotion_window = []
    emotion_prediction = []
    
    # Khởi tạo video
    video_capture = cv2.VideoCapture(0)

    
                                #---------------------------------------------#
                                # Tạo các layout được hiển thị trong phần mềm #
                                #---------------------------------------------#
                
    colcamera_layout = [[sg.Text("Camera",font=("Helvetica",15), size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="Camera")]] # Layout của camera
    colcamera = sg.Column(colcamera_layout, element_justification='center')

    colprobability_layout = [[sg.Text("Probability",font=("Helvetica",15), size=(60, 1), justification="center")],
                            [sg.Graph(GRAPH_SIZE, (0,-400), DATA_SIZE, key="Graph")]] # Layout của biểu đồ
    colprobability = sg.Column(colprobability_layout, element_justification='center')

    colslayout = [colcamera, colprobability] 
    layout = [[colslayout]] # Gộp các layout

    window = sg.Window("Facial Expression Recognition", layout,finalize=True, size=(1100,576)) # Khởi tạo cửa sổ window
    
    graph = window["Graph"] 

                                                #-------------------#
                                                # Chạy chương trình #
                                                #-------------------#    

    while True:
        event, values = window.read(timeout=20)

        if event == sg.WIN_CLOSED:
            break

                                                #---------------------------#
                                                # Dự đoán cảm xúc khuôn mặt #
                                                #---------------------------#    
        
        bgr_image = video_capture.read()[1]
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, gray_image)
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = bgr_image.copy()

        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)


            label = emotion_labels[emotion_prediction.argmax()]

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            color = np.asarray((0, 0, 255))
            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)
            
                                                    #-----------------------#
                                                    # Hiển thị lên màn hình #
                                                    #-----------------------#
                        
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        imgbytes = cv2.imencode(".png", bgr_image)[1].tobytes()
        window["Camera"].update(data=imgbytes)

                                                #-----------------------------#
                                                # Vẽ biểu đồ và tính xác suất #
                                                #-----------------------------#    
        
        graph.erase()
        for (i, (emotion, prob)) in enumerate(zip(emotion_labels, emotion_prediction[0])):
            graph_value = int(prob*300)     # Tính giá trị của cột
            pro_text = "{:.2f}%".format(prob * 100) # Tính xác suất các biểu cảm
            
            graph.draw_rectangle(top_left=(i * BAR_SPACING + EDGE_OFFSET, graph_value),
                                 bottom_right=(i * BAR_SPACING + EDGE_OFFSET + BAR_WIDTH, 0),
                                 fill_color='blue') # Vẽ biểu đồ
            graph.draw_text(text=pro_text, location=(i*BAR_SPACING+EDGE_OFFSET+15, graph_value+10), font='_ 8') # Ghi số liệu trên cột
            graph.draw_text(text=emotion, location=(i*BAR_SPACING+EDGE_OFFSET+10, -30), font='_ 8') # Ghi tên các biểu cảm