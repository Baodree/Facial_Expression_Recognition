import PySimpleGUI as sg
from Image_Function import *
from Camera_Function import *
from myfile.Util import *
                                #---------------------------------------------#
                                # Tạo các layout được hiển thị trong phần mềm #
                                #---------------------------------------------#
            
layout = [[sg.Button('Camera',pad=((0, 0), 40),size=(20,4))],
           [sg.Button('Image',pad=((0, 0), 20),size=(20,4))]]

window = sg.Window("Facial Expression Recognition", layout, size=(500, 300),finalize=True, element_justification='c')

                                                #-------------------#
                                                # Chạy chương trình #
                                                #-------------------#  

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == 'Camera':
        Camera_function()
    elif event == 'Image':
        Image_function()

window.close()