# import sys
# import cv2
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider


# class VideoPlayer(QMainWindow):
#     def __init__(self, filename):
#         super().__init__()

#         # Video capture object
#         self.capture = cv2.VideoCapture(filename)

#         # Get video properties
#         self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.frame_rate = int(self.capture.get(cv2.CAP_PROP_FPS))
#         self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         # Set up GUI
#         self.init_ui()

#         # Set up timer to update video
#         self.timer = self.startTimer(1000//self.frame_rate)

#     def init_ui(self):
#         # Create video widget
#         self.video_widget = QLabel(self)
#         self.setCentralWidget(self.video_widget)

#         # Create slider widget
#         self.slider = QSlider(Qt.Horizontal, self)
#         self.slider.setRange(0, self.frame_count)
#         self.slider.sliderMoved.connect(self.set_frame)

#         # Set window properties
#         self.setWindowTitle('Video Player')
#         self.setGeometry(100, 100, self.frame_width, self.frame_height+50)

#         # Show window
#         self.show()

#     def set_frame(self, frame_num):
#         # Set the current frame to the frame number selected on the slider
#         self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#         ret, frame = self.capture.read()

#         # Convert frame to QImage and display it on the video widget
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
#         pix = QPixmap.fromImage(img)
#         self.video_widget.setPixmap(pix)

#     def timerEvent(self, event):
#         # Update the slider and video widget with the next frame
#         frame_num = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
#         self.slider.setValue(frame_num)
#         ret, frame = self.capture.read()

#         # Convert frame to QImage and display it on the video widget
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
#         pix = QPixmap.fromImage(img)
#         self.video_widget.setPixmap(pix)

#     def closeEvent(self, event):
#         # Release the video capture object when the window is closed
#         self.capture.release()


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     player = VideoPlayer(r'D:\mediapipe\actionDetection\source\BaseballPitch\v_BaseballPitch_g01_c06.avi')
#     sys.exit(app.exec_())
# import cv2

# # Hàm xử lý sự kiện chuột
# def on_mouse(event, x, y, flags, param):
#     global start_time, end_time, frame_count
#     if event == cv2.EVENT_LBUTTONUP:
#         start_time = frame_count / fps
#     elif event == cv2.EVENT_LBUTTONUP:
#         end_time = frame_count / fps

# # Đường dẫn đến video cần cắt
# video_path = "WIN_20230220_22_07_52_Pro.mp4"

# # Tạo đối tượng đọc video
# cap = cv2.VideoCapture(video_path)

# # Lấy thông tin về video
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Tạo cửa sổ để hiển thị video
# cv2.namedWindow("Video")
# cv2.setMouseCallback("Video", on_mouse)

# # Bắt đầu đọc video
# start_time = 0
# end_time = frame_count / fps
# out = None
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         # Hiển thị frame trên cửa sổ
#         cv2.imshow("Video", frame)
#         # Ghi video đầu ra
#         if out is not None:
#             out.write(frame)
#         # Tính toán thời gian hiện tại của video
#         current_time = frame_count / fps
#         # Kết thúc video nếu hết thời gian cần cắt
#         if current_time > end_time:
#             break
#         # Cắt video nếu đã chọn thời gian cắt
#         if start_time < end_time:
#             if current_time >= start_time and current_time <= end_time:
#                 if out is None:
#                     out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
#         # Đợi sự kiện từ bàn phím
#         key = cv2.waitKey(100)
#         if key == 27: # Nhấn phím ESC để thoát
#             break
#     else:
#         break

# # Giải phóng bộ nhớ và đóng cửa sổ
# cap.release()
# if out is not None:
#     out.release()
# cv2.destroyAllWindows()


