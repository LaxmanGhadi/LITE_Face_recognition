# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout , QLineEdit
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtCore import QThread, pyqtSignal , Qt
# import cv2
# import sys
# import numpy as np
# import Face_recog_V2 
# import os 

# class Camera_Interface(QThread):
#     frame_update = pyqtSignal(QImage)
#     def __init__(self):
#         super().__init__()
#         self.running = False
#         self.cap = None
#         self.person = 'No person'

#     def run(self):
#         self.cap = cv2.VideoCapture(0)
#         self.running = True

#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
            
#             # Convert frame to RGB (from BGR)
#             rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             face = Face_recog_V2.detect_face(rgb_img)
#             if face is None:
#                 self.person = 'No person'
#             else:
#                 score,name= Face_recog_V2.check_person(face)
#                 if score*100 > 70:
                    
#                     self.person = name
#                     # cv2.putText(rgb_img, name +str(score*100)+"%",(50,50),cv2.FONT_HERSHEY_DUPLEX , 1, (245, 66, 111), 2, cv2.LINE_AA)
#                 else:
                    
#                     self.person= 'Unknown person'
#             h, w, ch = rgb_img.shape  # Get height, width, and number of channels
#             bytes_per_line = ch * w  # Calculate bytes per line (width * channels)
            
#             # Convert the image to bytes
#             byte_data = rgb_img.tobytes()
            
#             # Create QImage from raw byte data
#             qt_img = QImage(byte_data, w, h, bytes_per_line, QImage.Format_RGB888)
            
#             # Emit the signal with the updated frame
#             self.frame_update.emit(qt_img)

#         self.cap.release()

#     def stop(self):
#         self.running = False
#         self.wait()

# class Window(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.inst = 'Enter the name of the perosn '
#         self.save_path = ''
#         self.setWindowTitle("OpenCV + PyQt5")
#         # self.resize(900, 600)
#         self.image_no = 1
#         self.label2 = QLabel("No person")
#         self.label2.setAlignment(Qt.AlignHCenter)
#         self.label3 = QLabel(self.inst)
#         self.label3.setAlignment(Qt.AlignHCenter)
#         self.label3.hide()

#         # Setup the label for displaying the camera feed
#         self.label = QLabel("Press Start to see camera Feed")
#         self.label.setStyleSheet("background-color:black;color:white")
#         self.label.setAlignment(Qt.AlignHCenter)
#         self.label.setFixedSize(640, 480)
#         self.editBox = QLineEdit(self)
#         self.editBox.setPlaceholderText("Enter name of the person")
#         self.editBox.hide()
#         self.editBox.setEnabled(False)
#         # Setup buttons
#         self.btn_start = QPushButton("Start")
#         self.btn_stop = QPushButton("Stop")
#         self.btn_AdFc = QPushButton("Add Face")
#         self.btn_MrAt = QPushButton("Mark Attendance")
#         self.Clk_pic = QPushButton("Click Photo")
#         self.Clk_pic.hide()
#         self.Crt_fld = QPushButton("Create")
#         self.Crt_fld.hide()
#         self.Clk_pic.setEnabled(False)

#         # Layout setup
#         h_layout = QHBoxLayout()
#         h_layout.addWidget(self.btn_start)
#         h_layout.addWidget(self.btn_stop)
#         h_layout.addWidget(self.btn_AdFc)
#         h_layout.addWidget(self.btn_MrAt)


#         h_layout2 = QHBoxLayout() 
#         h_layout2.addWidget(self.editBox)
#         h_layout2.addWidget(self.Crt_fld)

#         v_layout = QVBoxLayout()
#         v_layout.addWidget(self.label)
#         v_layout.addWidget(self.label2)
#         v_layout.addWidget(self.label3)
#         v_layout.addLayout(h_layout2)
#         v_layout.addWidget(self.Clk_pic)
#         v_layout.addLayout(h_layout)

#         self.setLayout(v_layout)
        
#         # Camera thread
#         self.thread = Camera_Interface()

#         # Button signals
#         self.btn_start.clicked.connect(self.start_camera)
#         self.btn_stop.clicked.connect(self.stop_camera)
#         self.btn_AdFc.clicked.connect(self.Add_Face)
#         self.btn_MrAt.clicked.connect(self.Mark_Attendance)
#         self.Crt_fld.clicked.connect(self.Create_folder)
#         self.Clk_pic.clicked.connect(self.Take_pic)
#         # Connect the camera feed signal
#         self.thread.frame_update.connect(self.update_frame)

#     def start_camera(self):
#         if not self.thread.isRunning():
#             self.thread.start()

#     def stop_camera(self):
#         if self.thread.isRunning():
#             self.thread.stop()

#     def Add_Face(self):
#         self.editBox.setEnabled(True)
#         self.Crt_fld.setEnabled(True)
#         self.editBox.show()
#         self.Crt_fld.show()
#         self.label3.show()
#         self.Clk_pic.show()
#         if not self.thread.isRunning():
#             self.thread.start()

#     def Mark_Attendance(self):
#         print("attendance marked")

#     def update_frame(self, image):
#         self.label.setPixmap(QPixmap.fromImage(image))
#         self.label2.setText(self.thread.person)
    
#     def Create_folder(self):
#         txt = self.editBox.text()
#         if len(txt)>0:
#           self.save_path = os.path.join('Dataset/People',self.editBox.text())
#           os.makedirs(self.save_path, exist_ok=True)
#           self.Clk_pic.setEnabled(True)
        
#     def Take_pic(self, image):
#         img_name = os.path.join(self.save_path,f"{self.editBox.text()}_{self.image_no}.jpg")
#         cv2.imwrite(img_name,image)
#         self.label3.setText(f"{self.image_no} images have been saved")
#         self.image_no +=1
# # Run the application
# app = QApplication(sys.argv)
# win = Window()
# win.show()
# sys.exit(app.exec_())

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import cv2
import sys
import os
import Face_recog_V2
import All_Face_Encod

class Camera_Interface(QThread):
    frame_update = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        self.person = 'No person'

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert frame to RGB (from BGR)
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = Face_recog_V2.detect_face(rgb_img)
            if face is None:
                self.person = 'No person'
            else:
                score, name = Face_recog_V2.check_person(face)
                if score * 100 > 70:
                    self.person = name
                else:
                    self.person = 'Unknown person'

            h, w, ch = rgb_img.shape  # Get height, width, and number of channels
            bytes_per_line = ch * w  # Calculate bytes per line (width * channels)

            # Convert the image to bytes
            byte_data = rgb_img.tobytes()

            # Create QImage from raw byte data
            qt_img = QImage(byte_data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Emit the signal with the updated frame
            self.frame_update.emit(qt_img)

        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.inst = 'Enter the name of the person '
        self.save_path = ''
        self.setWindowTitle("OpenCV + PyQt5")
        self.image_no = 1
        self.label2 = QLabel("No person")
        self.label2.setAlignment(Qt.AlignHCenter)
        self.label3 = QLabel(self.inst)
        self.label3.setAlignment(Qt.AlignHCenter)
        self.label3.hide()

        # Setup the label for displaying the camera feed
        self.label = QLabel("Press Start to see camera Feed")
        self.label.setStyleSheet("background-color:black;color:white")
        self.label.setAlignment(Qt.AlignHCenter)
        self.label.setFixedSize(640, 480)
        self.editBox = QLineEdit(self)
        self.editBox.setPlaceholderText("Enter name of the person")
        self.editBox.hide()
        self.editBox.setEnabled(False)
        # Setup buttons
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_AdFc = QPushButton("Add Face")
        self.btn_MrAt = QPushButton("Mark Attendance")
        self.Clk_pic = QPushButton("Click Photo")
        self.Clk_pic.hide()
        self.Crt_fld = QPushButton("Create")
        self.Crt_fld.hide()
        self.Clk_pic.setEnabled(False)
        self.Sav_img = QPushButton("Save")
        self.Sav_img.hide()
        self.Sav_img.setEnabled(False)

        # Layout setup
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.btn_start)
        h_layout.addWidget(self.btn_stop)
        h_layout.addWidget(self.btn_AdFc)
        h_layout.addWidget(self.btn_MrAt)

        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(self.editBox)
        h_layout2.addWidget(self.Crt_fld)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.label)
        v_layout.addWidget(self.label2)
        v_layout.addWidget(self.label3)
        v_layout.addLayout(h_layout2)
        v_layout.addWidget(self.Clk_pic)
        v_layout.addWidget(self.Sav_img)
        v_layout.addLayout(h_layout)

        self.setLayout(v_layout)

        # Camera thread
        self.thread = Camera_Interface()

        # Button signals
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_AdFc.clicked.connect(self.Add_Face)
        self.btn_MrAt.clicked.connect(self.Mark_Attendance)
        self.Crt_fld.clicked.connect(self.Create_folder)
        self.Clk_pic.clicked.connect(self.Take_pic)
        self.Sav_img.clicked.connect(self.Encode)

        # Connect the camera feed signal
        self.thread.frame_update.connect(self.update_frame)

    def start_camera(self):
        if not self.thread.isRunning():
            self.thread.start()

    def stop_camera(self):
        if self.thread.isRunning():
            self.thread.stop()

    def Add_Face(self):
        self.editBox.setEnabled(True)
        self.Crt_fld.setEnabled(True)
        self.Sav_img.show()
        self.editBox.show()
        self.Crt_fld.show()
        self.label3.show()
        self.Clk_pic.show()
        if not self.thread.isRunning():
            self.thread.start()

    def Mark_Attendance(self):
        print("attendance marked")

    def update_frame(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
        self.label2.setText(self.thread.person)

    def Create_folder(self):
        txt = self.editBox.text()
        if len(txt) > 0:
            self.save_path = os.path.join('Dataset/People', self.editBox.text())
            os.makedirs(self.save_path, exist_ok=True)
            self.Clk_pic.setEnabled(True)

    def Take_pic(self):
        if self.thread.cap is not None:
            ret, frame = self.thread.cap.read()
            if ret:
                img_name = os.path.join(self.save_path, f"{self.editBox.text()}_{self.image_no}.jpg")
                cv2.imwrite(img_name, frame)
                self.label3.setText(f"{self.image_no} images have been saved")
                self.image_no += 1
        if self.image_no>=6:
            self.Sav_img.setEnabled(True)
    
    def Encode(self):
        All_Face_Encod.create_embedding_database()
        self.thread.stop()
        self.thread.start()
# Run the application
app = QApplication(sys.argv)
win = Window()
win.show()
sys.exit(app.exec_())
