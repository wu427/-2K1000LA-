import numpy as np
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5 import QtMultimedia

import threading
from PyQt5.QtMultimedia import QMediaPlayer
import time

import matplotlib
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import socket
import pyaudio

import sys
import sqlite3
import hashlib
from PyQt5 import QtCore, QtGui, QtWidgets

import os
import datetime
from PIL import ImageQt
# 音频设置
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

class VideoServer1(QtCore.QObject):
    update_image = QtCore.pyqtSignal(QtGui.QImage)
    update_classes = QtCore.pyqtSignal(list)
    update_centers = QtCore.pyqtSignal(list)  # 新增信号以更新中心点
    update_status = QtCore.pyqtSignal(str)

    def __init__(self, ip, port):
        super().__init__()
        self.ip = ip
        self.port = port
        self.is_running = False
        self.audio_stream = None
        self.pyaudio_instance = None
    def start_audio_playback(self):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio_instance.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True
        )

    def stop_audio_playback(self):
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()

    def start(self):
        self.is_running = True
        threading.Thread(target=self.run_server, daemon=True).start()

    def draw_centers_on_image(self, image, centers):
        for center in centers:
            cv2.circle(image, (center[0], center[1]), radius=5, color=(0, 255, 0), thickness=-1)  # 绿色圆点
        return image

    def run_server(self):
        global image_data, center_points
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.ip, self.port))
        server_socket.listen(1)
        print(f"服务器已启动，等待连接于 {self.ip}:{self.port}...")
        conn, addr = server_socket.accept()
        print(f"连接来自: {addr}")

        # 连接建立后再初始化音频
        self.start_audio_playback()

        while self.is_running:
            try:
                # 接收数据类型标识 (0=视频, 1=音频)
                data_type = conn.recv(1)
                if not data_type:
                    print("连接已关闭")
                    break

                if data_type == b'a':  # 视频数据
                    # 接收类别数据
                    class_data = b""
                    while True:
                        packet = conn.recv(1)
                        if not packet:
                            print("未接收到类别数据，连接可能已关闭")
                            return
                        class_data += packet
                        if b'\n' in class_data:  # 假设类别数据以换行符结束
                            break

                    if class_data:
                        newline_index = class_data.find(b'\n')
                        image_data = class_data[newline_index + 1:]  # 只将换行符之后的数据存储到 image_data
                        class_data = class_data[:newline_index + 1]  # 包含换行符

                        try:
                            class_data = class_data.decode('utf-8').strip()  # 解码为UTF-8
                            class_names = class_data.split(',')
                            self.update_classes.emit(class_names)
                            print(f"接收到类别数据: {class_names}")

                            # 接收中心点数据
                            center_data = b""
                            while True:
                                packet = conn.recv(1)
                                if not packet:
                                    print("未接收到中心点数据，连接可能已关闭")
                                    return
                                center_data += packet
                                if b'\n' in center_data:  # 假设中心点数据以换行符结束
                                    break

                            center_data = center_data.decode('utf-8').strip()

                            if center_data:
                                # 解析中心点数据
                                center_points = []
                                for point in center_data.split(';'):
                                    if point:  # 确保不处理空字符串
                                        x, y = map(int, point.split(','))
                                        center_points.append([x, y])  # 将每个点作为一个数组添加
                                self.update_centers.emit(center_points)
                                #print(f"接收到中心点数据: {center_points}")


                            size_data = b""
                            while len(size_data) < 4:  # 需要接收4个字节
                                packet = conn.recv(1)
                                if not packet:
                                    print("未接收到图像数据长度，连接可能已关闭")
                                    return
                                size_data += packet

                            # 将接收到的字节转换为整数
                            image_size = int.from_bytes(size_data, byteorder='big')
                            #print(f"接收到图像大小: {image_size}   {size_data}")

                            frame_data = bytearray()

                            while len(frame_data) < image_size:
                                packet = conn.recv(65507)
                                if not packet:
                                    print("未接收到图像数据，连接可能已关闭")
                                    return
                                frame_data.extend(packet)
                            #print(f"接收到frame_data数据: {frame_data}")

                            if frame_data:
                                np_array = np.frombuffer(frame_data, np.uint8)
                                frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                                if frame is not None:
                                    #print("成功接收到图像数据")
                                    # 绘制中心点
                                    if 'center_points' in locals():  # 确保中心点数据存在

                                        self.update_centers.emit(center_points)
                                        print("1")
                                        qimg = self.convert_to_qimage(frame)
                                        self.update_image.emit(qimg)
                                    else:
                                        qimg = self.convert_to_qimage(frame)
                                        self.update_image.emit(qimg)
                                else:
                                    print("图像解码失败")

                        except UnicodeDecodeError:
                            print("类别数据解码失败，可能不是有效的UTF-8字符串")

                elif data_type == b'b':  # 音频数据
                    # 接收音频数据长度
                    audio_size_bytes = conn.recv(4)
                    if not audio_size_bytes:
                        print("未接收到音频数据长度，连接可能已关闭")
                        break
                    audio_size = int.from_bytes(audio_size_bytes, byteorder='big')
                    #print(f"接收到音频数据长度: {audio_size} bytes")  # 调试输出

                    # 接收音频数据
                    audio_data = bytearray()
                    while len(audio_data) < audio_size:
                        packet = conn.recv(min(65507, audio_size - len(audio_data)))
                        if not packet:
                            print("未接收到音频数据，连接可能已关闭")
                            break
                        audio_data.extend(packet)

                    if len(audio_data) != audio_size:
                        print(f"接收到的音频数据大小不匹配: 预期 {audio_size}, 实际 {len(audio_data)}")
                    else:
                        #print("成功接收到音频数据")
                        self.audio_stream.write(bytes(audio_data))  # 确保将bytearray转换为bytes

            except Exception as e:
                print(f"接收数据时发生错误: {e}")
                break

        conn.close()
        server_socket.close()
        self.stop_audio_playback()

    def convert_to_qimage(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        return QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_BGR888)


class DottedLine(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super(DottedLine, self).__init__(parent)
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        pen = QtGui.QPen(QtCore.Qt.white, 2)
        pen.setDashPattern([4, 4])
        painter.setPen(pen)
        painter.drawLine(0, 0, self.width(), 0)


class Ui_Form(object):
    def setupUi(self, Form):
        self.Form = Form
        Form.setObjectName("Form")
        Form.resize(300, 400)

        # Background
        self.background_label = QtWidgets.QLabel(Form)
        self.background_label.setGeometry(QtCore.QRect(0, 0, 300, 400))
        self.background_label.setPixmap(QtGui.QPixmap("登录界面.png"))  # Replace with your image path
        self.background_label.setScaledContents(True)
        self.background_label.setObjectName("background_label")

        # Semi-transparent panel
        self.small_background2 = QtWidgets.QLabel(Form)
        self.small_background2.setGeometry(QtCore.QRect(10, 70, 280, 340))
        self.small_background2.setStyleSheet("background-color: rgba(30, 31, 34, 120); border-radius: 10px;")
        self.small_background2.setObjectName("small_background")

        # Title
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setGeometry(QtCore.QRect(20, 70, 260, 50))
        self.label_7.setObjectName("label_7")
        default_font = self.label_7.font()
        default_font.setPointSize(12)
        self.label_7.setFont(default_font)

        # Username field
        self.username_edit = QtWidgets.QLineEdit(Form)
        self.username_edit.setGeometry(QtCore.QRect(20, 140, 260, 30))
        self.username_edit.setObjectName("username_edit")
        self.username_edit.setStyleSheet(
            """background: transparent;border: none; border-bottom: 1px solid white; color: white; padding-bottom: 5px;""")
        self.username_edit.setPlaceholderText("请输入用户名")

        # Password field
        self.password_edit = QtWidgets.QLineEdit(Form)
        self.password_edit.setGeometry(QtCore.QRect(20, 190, 260, 30))
        self.password_edit.setObjectName("password_edit")
        self.password_edit.setStyleSheet(
            """background: transparent;border: none;border-bottom: 1px solid white;color: white;padding-bottom: 5px;""")
        self.password_edit.setPlaceholderText("请输入密码")
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)

        # Login button
        self.login_button = QtWidgets.QPushButton(Form)
        button_width = 240
        button_height = 30
        button_x = int((300 - button_width) / 2)
        self.login_button.setGeometry(QtCore.QRect(button_x, 250, button_width, button_height))
        self.login_button.setObjectName("login_button")
        self.login_button.setStyleSheet(
            """background-color: rgba(6, 187, 252, 200);color: white;border: 1px solid gray;border-radius: 7px;""")
        self.login_button.clicked.connect(self.login)

        # Register link
        self.register_label = QtWidgets.QLabel(Form)
        self.register_label.setGeometry(QtCore.QRect(0, 300, 300, 20))
        self.register_label.setAlignment(QtCore.Qt.AlignCenter)
        self.register_label.setStyleSheet("color: white;")
        self.register_label.setText('<a href="#" style="color: white; text-decoration: none;">没有账号？立即注册</a>')
        self.register_label.setOpenExternalLinks(False)
        self.register_label.linkActivated.connect(self.show_register)

        # Bring all elements to front
        self.background_label.raise_()
        self.small_background2.raise_()
        self.username_edit.raise_()
        self.password_edit.raise_()
        self.label_7.raise_()
        self.login_button.raise_()
        self.register_label.raise_()

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "“洞若观火”——火灾检测系统"))
        self.label_7.setText(_translate("Form",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:700; color:white;\">登录</span></p></body></html>"))
        self.login_button.setText(_translate("Form", "登录"))

    def login(self):
        username = self.username_edit.text()
        password = self.password_edit.text()

        if not username or not password:
            QtWidgets.QMessageBox.critical(self.Form, "错误", "请输入用户名和密码！")
            return

        db = DatabaseManager()
        result = db.verify_user(username, password)
        db.close()

        if result:
            ip, port = result  # 正确解包返回的元组

            try:
                # 正确创建主窗口实例
                self.main_app = APP1(ip=ip, port=port)  # 明确传递参数
                self.main_app.Form.show()  # 显示主窗口
                self.Form.close()  # 关闭登录窗口
            except Exception as e:
                QtWidgets.QMessageBox.critical(self.Form, "错误", f"启动失败: {str(e)}")
        else:
            QtWidgets.QMessageBox.critical(self.Form, "错误", "用户名或密码错误！")


    def show_register(self):
        self.register_window = QtWidgets.QWidget()
        self.register_ui = Ui_RegisterForm()
        self.register_ui.setupUi(self.register_window)
        self.register_window.show()
        self.Form.hide()





# ====================== 数据库管理类 ======================
class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect("user_data.db")
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """创建用户表（如果不存在）"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                ip TEXT NOT NULL,
                port INTEGER NOT NULL
            )
        """)
        self.conn.commit()

    def register_user(self, username, password, ip, port):
        """注册用户，密码用 SHA-256 加密"""
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        try:
            self.cursor.execute(
                "INSERT INTO users (username, password, ip, port) VALUES (?, ?, ?, ?)",
                (username, hashed_password, ip, port)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"注册错误: {e}")
            return False

    def verify_user(self, username, password):
        """验证用户名和密码，返回 (ip, port) 或 None"""
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        self.cursor.execute(
            "SELECT ip, port FROM users WHERE username = ? AND password = ?",
            (username, hashed_password)
        )
        return self.cursor.fetchone()  # 返回 (ip, port) 或 None

    def close(self):
        """关闭数据库连接"""
        self.conn.close()


# ====================== 注册界面 ======================
class Ui_RegisterForm(object):
    def setupUi(self, Form):
        self.Form = Form
        Form.setObjectName("Form")
        Form.resize(300, 450)

        # Background
        self.background_label = QtWidgets.QLabel(Form)
        self.background_label.setGeometry(QtCore.QRect(0, 0, 300, 450))
        self.background_label.setPixmap(QtGui.QPixmap("登录界面.png"))  # 替换为你的背景图路径
        self.background_label.setScaledContents(True)
        self.background_label.setObjectName("background_label")

        # Semi-transparent panel
        self.small_background2 = QtWidgets.QLabel(Form)
        self.small_background2.setGeometry(QtCore.QRect(10, 50, 280, 380))
        self.small_background2.setStyleSheet("background-color: rgba(30, 31, 34, 120); border-radius: 10px;")
        self.small_background2.setObjectName("small_background")

        # Title
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setGeometry(QtCore.QRect(20, 50, 260, 50))
        self.label_7.setObjectName("label_7")
        default_font = self.label_7.font()
        default_font.setPointSize(12)
        self.label_7.setFont(default_font)

        # Username field
        self.username_edit = QtWidgets.QLineEdit(Form)
        self.username_edit.setGeometry(QtCore.QRect(20, 120, 260, 30))
        self.username_edit.setObjectName("username_edit")
        self.username_edit.setStyleSheet(
            "background: transparent; border: none; border-bottom: 1px solid white; color: white; padding-bottom: 5px;")
        self.username_edit.setPlaceholderText("请输入用户名")

        # Password field
        self.password_edit = QtWidgets.QLineEdit(Form)
        self.password_edit.setGeometry(QtCore.QRect(20, 170, 260, 30))
        self.password_edit.setObjectName("password_edit")
        self.password_edit.setStyleSheet(
            "background: transparent; border: none; border-bottom: 1px solid white; color: white; padding-bottom: 5px;")
        self.password_edit.setPlaceholderText("请输入密码")
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)

        # Confirm Password field
        self.confirm_password_edit = QtWidgets.QLineEdit(Form)
        self.confirm_password_edit.setGeometry(QtCore.QRect(20, 220, 260, 30))
        self.confirm_password_edit.setObjectName("confirm_password_edit")
        self.confirm_password_edit.setStyleSheet(
            "background: transparent; border: none; border-bottom: 1px solid white; color: white; padding-bottom: 5px;")
        self.confirm_password_edit.setPlaceholderText("请确认密码")
        self.confirm_password_edit.setEchoMode(QtWidgets.QLineEdit.Password)

        # IP field
        self.ip_edit = QtWidgets.QLineEdit(Form)
        self.ip_edit.setGeometry(QtCore.QRect(20, 270, 260, 30))
        self.ip_edit.setObjectName("ip_edit")
        self.ip_edit.setStyleSheet(
            "background: transparent; border: none; border-bottom: 1px solid white; color: white; padding-bottom: 5px;")
        self.ip_edit.setPlaceholderText("请输入IP地址")

        # Port field
        self.port_edit = QtWidgets.QLineEdit(Form)
        self.port_edit.setGeometry(QtCore.QRect(20, 320, 260, 30))
        self.port_edit.setObjectName("port_edit")
        self.port_edit.setStyleSheet(
            "background: transparent; border: none; border-bottom: 1px solid white; color: white; padding-bottom: 5px;")
        self.port_edit.setPlaceholderText("请输入端口")

        # Register button
        self.register_button = QtWidgets.QPushButton(Form)
        button_width = 240
        button_height = 30
        button_x = int((300 - button_width) / 2)
        self.register_button.setGeometry(QtCore.QRect(button_x, 370, button_width, button_height))
        self.register_button.setObjectName("register_button")
        self.register_button.setStyleSheet(
            "background-color: rgba(6, 187, 252, 200); color: white; border: 1px solid gray; border-radius: 7px;")
        self.register_button.clicked.connect(self.register)

        # # 添加返回按钮
        # self.back_button = QtWidgets.QPushButton(Form)
        # self.back_button.setGeometry(QtCore.QRect(button_x, 410, button_width, button_height))
        # self.back_button.setObjectName("back_button")
        # self.back_button.setStyleSheet(
        #     "background-color: rgba(255, 255, 255, 100); color: white; border: 1px solid gray; border-radius: 7px;")
        # self.back_button.clicked.connect(self.back_to_login)
        # self.back_button.setText("返回登录")
        # 返回按钮 (左上角箭头图标)
        self.back_button = QtWidgets.QPushButton(Form)
        self.back_button.setGeometry(QtCore.QRect(20, 60, 30, 30))  # 左上角位置
        self.back_button.setObjectName("back_button")
        self.back_button.setStyleSheet("""
                   QPushButton {
                       background: transparent;
                       border: none;
                       color: white;
                   }
                   QPushButton:hover {
                       color: #06BBFC;
                   }
               """)
        # 设置箭头图标 (使用Unicode箭头字符)
        self.back_button.setText("←")  # 左箭头符号
        self.back_button.setFont(QtGui.QFont("Arial", 16))  # 增大箭头大小
        self.back_button.clicked.connect(self.back_to_login)

        # Bring all elements to front
        self.background_label.raise_()
        self.small_background2.raise_()
        self.username_edit.raise_()
        self.password_edit.raise_()
        self.confirm_password_edit.raise_()
        self.ip_edit.raise_()
        self.port_edit.raise_()
        self.label_7.raise_()
        self.register_button.raise_()
        self.back_button.raise_()

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "注册账号"))
        self.label_7.setText(_translate("Form",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:700; color:white;\">注册</span></p></body></html>"))
        self.register_button.setText(_translate("Form", "注册"))

    def register(self):
        username = self.username_edit.text()
        password = self.password_edit.text()
        confirm_password = self.confirm_password_edit.text()
        ip = self.ip_edit.text()
        port = self.port_edit.text()

        if not all([username, password, confirm_password, ip, port]):
            QtWidgets.QMessageBox.critical(self.Form, "错误", "请填写所有字段！")
            return

        if password != confirm_password:
            QtWidgets.QMessageBox.critical(self.Form, "错误", "两次输入的密码不一致！")
            return

        try:
            port = int(port)
        except ValueError:
            QtWidgets.QMessageBox.critical(self.Form, "错误", "端口必须是一个整数！")
            return

        db = DatabaseManager()
        if db.register_user(username, password, ip, port):
            QtWidgets.QMessageBox.information(self.Form, "成功", "注册成功！")
            #self.back_to_login()  # 注册成功后返回登录界面
            #self.Form.close()
        else:
            QtWidgets.QMessageBox.warning(self.Form, "错误", "用户名已存在！")
        db.close()

    def back_to_login(self):
        """返回登录界面"""
        self.login_window = QtWidgets.QWidget()
        self.login_ui = Ui_Form()
        self.login_ui.setupUi(self.login_window)
        self.login_window.show()
        self.Form.close()
class APP1(object):
    update_image = QtCore.pyqtSignal(QtGui.QImage)
    update_centers = QtCore.pyqtSignal(list)  # 添加 update_centers 信号
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.video_server = VideoServer1(self.ip, self.port)
        self.detected_classes = set()
        self.is_running = False
        self.last_detection_time = None
        self.setupUi()
        self.movie = None
        self.gif_path = None
        self.player = QMediaPlayer()
        self.is_alarm_playing = False
        self.last_cleared_time = QtCore.QDateTime.currentDateTime()
        self.last_cleared_time2 = QtCore.QDateTime.currentDateTime()
        self.video_server.update_image.connect(self.update_image)
        self.video_server.update_classes.connect(self.update_classes)
        self.video_server.update_status.connect(self.update_status)
        self.video_server.update_centers.connect(self.update_centers)
        self.center_points = []

        self.timer_check_detection = QtCore.QTimer()
        self.timer_check_detection.timeout.connect(self.check_detection_status)
        self.timer_check_detection.start(10000)

        self.frame_counter = 0
        self.frame_skip_interval = 2
        self.latest_image = None
        self.latest_image_processed = False
        self.label.mousePressEvent = self.on_image_click
        self.current_mode = "normal"  # 初始模式为正常模式
        self.mode_changed = False  # 标志位，记录是否需要切换模式
        self.last_click_time = 0  # 记录上次点击的时间
        self.double_click_timer = QTimer()
        self.double_click_timer.setSingleShot(True)
        self.double_click_timer.timeout.connect(self.handle_single_click)

        # 深度估计相关设置
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.encoder = 'vits'  # Choose your encoder here
        self.depth_anything = DepthAnythingV2(**self.model_configs[self.encoder])
        self.depth_anything.load_state_dict(
            torch.load(f'checkpoints/depth_anything_v2_{self.encoder}.pth', map_location='cpu'))
        self.depth_anything = self.depth_anything.to(self.device).eval()
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')

        self.lock = threading.Lock()  # 创建一个锁对象

        self.frame_counter = 0  # 新增的帧计数器
        self.frame_skip_interval = 2  # 设置跳帧间隔为2

        self.fire_save_dir = r"C:\\Users\\19736\\Desktop\\FireDetections"
        os.makedirs(self.fire_save_dir, exist_ok=True)
    def setupUi(self):
        self.Form = QtWidgets.QWidget()
        self.Form.setObjectName("Form")
        self.Form.resize(1250, 701)
        width_ratio = 1250 / 1291
        height_ratio = 701 / 721
        self.listView = QtWidgets.QListView(self.Form)
        self.listView.setGeometry(QtCore.QRect(0, 0, 1250, 701))
        self.listView.setStyleSheet("border-image: url(主界面.png);")
        self.listView.setObjectName("listView")
        self.label = QtWidgets.QLabel(self.Form)
        self.label.setGeometry(QtCore.QRect(int(395 * width_ratio), int(140 * height_ratio), int(505 * width_ratio), int(420 * height_ratio)))
        self.label.setObjectName("label")
        self.label.setStyleSheet("background: transparent; color: white;")

        # 设置点击事件
        self.label.mousePressEvent = self.on_image_click  # 添加鼠标点击事件

        # self.label1 = QtWidgets.QLabel(self.Form)
        # self.label1.setGeometry(QtCore.QRect(int(395 * width_ratio), int(140 * height_ratio), int(505 * width_ratio),
        #                                     int(420 * height_ratio)))
        # self.label1.setObjectName("label1")
        # self.label1.setStyleSheet("background: transparent; color: white;")

        self.textEdit = QtWidgets.QTextEdit(self.Form)
        self.textEdit.setGeometry(QtCore.QRect(int(16 * width_ratio), int(143 * height_ratio), int(327 * width_ratio), int(454 * height_ratio)))
        self.textEdit.setObjectName("textEdit")
        self.textEdit.setStyleSheet("background: transparent; color: white; border: none;")
        self.textEdit.setReadOnly(True)
        self.textEdit_2 = QtWidgets.QTextEdit(self.Form)
        self.textEdit_2.setGeometry(QtCore.QRect(int(933 * width_ratio), int(138 * height_ratio), int(367 * width_ratio), int(173 * height_ratio)))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_2.setStyleSheet("background: transparent; color: white; border: none;")
        self.textEdit_2.setReadOnly(True)
        self.gif_label = QtWidgets.QLabel(self.textEdit_2)
        self.gif_label.setGeometry(20, 0, 300, 180)
        self.gif_label.setScaledContents(True)
        self.textEdit_3 = QtWidgets.QTextEdit(self.Form)
        self.textEdit_3.setGeometry(QtCore.QRect(int(944 * width_ratio), int(433* height_ratio), int(328 * width_ratio), int(171 * height_ratio)))
        self.textEdit_3.setObjectName("textEdit_3")
        self.textEdit_3.setStyleSheet("background: transparent; color: white; border: none;")
        self.pushButton = QtWidgets.QPushButton(self.Form)
        self.pushButton.setGeometry(QtCore.QRect(int(430 * width_ratio), int(630 * height_ratio), int(151 * width_ratio), int(51 * height_ratio)))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setStyleSheet("background: rgba(6, 187, 252, 200); ""color: white; ""border-radius: 15px; ""font-size: 22px; ""font-family: YouSheBiaoTiYuan;")
        self.pushButton.setText("开始视频")
        self.pushButton.clicked.connect(self.start_video)
        self.pushButton_2 = QtWidgets.QPushButton(self.Form)
        self.pushButton_2.setGeometry(QtCore.QRect(int(700 * width_ratio), int(630 * height_ratio), int(151 * width_ratio), int(51 * height_ratio)))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setStyleSheet("background: rgba(6, 187, 252, 200); ""color: white; ""border-radius: 15px; ""font-size: 22px; ""font-family: YouSheBiaoTiYuan;")
        self.pushButton_2.setText("关闭视频")
        self.pushButton_2.clicked.connect(self.stop_video)
        self.time_label = QtWidgets.QLabel(self.Form)
        self.time_label.setGeometry(QtCore.QRect(int(1075 * width_ratio), int(5 * height_ratio), int(270 * width_ratio), int(51 * height_ratio)))
        self.time_label.setStyleSheet("font-size: 18px; color: black; background: transparent;")
        self.timer = QtCore.QTimer(self.Form)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        self.update_time()


        self.Form.setWindowTitle("洞若观“火”——工业火灾检测系统")
        self.Form.show()

    def start_video(self):
        self.video_server.start()
        self.label.show()
        self.gif_label.show()
        self.textEdit_3.show()
        self.textEdit.show()
        print("视频服务器已启动。")
    def stop_video(self):
        self.label.hide()
        self.gif_label.hide()
        self.textEdit_3.hide()
        self.textEdit.hide()
        print("视频服务器已停止。")



    @staticmethod
    def qimage2numpy(qimage, dtype='array'):

        result_shape = (qimage.height(), qimage.width())
        temp_shape = (qimage.height(), qimage.bytesPerLine() * 8 // qimage.depth())

        if qimage.format() in (QtGui.QImage.Format_ARGB32_Premultiplied,
                               QtGui.QImage.Format_ARGB32,
                               QtGui.QImage.Format_RGB32):
            if dtype == 'rec':
                dtype = QtGui.bgra_dtype
            elif dtype == 'array':
                dtype = np.uint8
                result_shape += (4,)
                temp_shape += (4,)
        elif qimage.format() == QtGui.QImage.Format_Indexed8:
            dtype = np.uint8
        else:
            raise ValueError("qimage2numpy only supports 32bit and 8bit images")

        buf = qimage.bits().asstring(qimage.byteCount())  # 使用 byteCount() 替代 numBytes()
        result = np.frombuffer(buf, dtype).reshape(temp_shape)

        if result_shape != temp_shape:
            result = result[:, :result_shape[1]]

        if qimage.format() == QtGui.QImage.Format_RGB32 and dtype == np.uint8:
            result = result[..., :3]

        result = result[:, :, ::-1]  # Convert BGR to RGB
        print(f"Converted array shape: {result.shape}")
        return result


    def update_centers(self, center_points):
        self.center_points = center_points


    def draw_centers_on_image(self, image, centers):
        for center in centers:
            x, y = center
            depth_value = 1  # 获取中心点处的深度值
            text = f"Depth: {depth_value:.2f}"
            image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            image = cv2.putText(image, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                                cv2.LINE_AA)
        return image


    def update_image(self, qimg):
        with self.lock:
            if qimg.isNull():
                print("接收到的图像为空，无法处理")
                return

            # 检查并转换 QImage 格式
            if qimg.format() not in (QtGui.QImage.Format_ARGB32,
                                     QtGui.QImage.Format_ARGB32_Premultiplied,
                                     QtGui.QImage.Format_RGB32,
                                     QtGui.QImage.Format_Indexed8):
                qimg = qimg.convertToFormat(QtGui.QImage.Format_ARGB32)

            self.latest_image = qimg
            self.latest_image_processed = False
            self.frame_counter += 1
            #print("2")
            if self.current_mode == "normal":
                # 正常模式下直接显示原始图像
                #print("normal")
                self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))
            elif self.current_mode == "gray":
                print("gray")
                # 如果当前帧不需要处理，则直接返回
                if self.frame_counter % self.frame_skip_interval != 0:
                    return
                # 启动深度估计线程
                threading.Thread(target=self.estimate_depth, args=(qimg,)).start()
            elif self.current_mode == "point":
                print("point")
                # # 点模式下显示原始图像并标出中心点深度值
                if self.frame_counter % 5 != 0:
                    return
                # 启动深度估计线程
                threading.Thread(target=self.estimate_depth2, args=(qimg,self.center_points)).start()



    @staticmethod
    def qimage2numpy2(qimage):
        """Convert QImage to numpy array with correct color channels"""
        # 强制转换为RGB888格式
        if qimage.format() != QtGui.QImage.Format_RGB888:
            qimage = qimage.convertToFormat(QtGui.QImage.Format_RGB888)

        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())

        # 转换为numpy数组 (height, width, 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))

        # OpenCV使用BGR顺序，所以需要转换
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        print(f"Converted array sample (BGR):\n{arr[240:245, 320:325]}")
        return arr

    def convert_to_qimage(self, frame):
        """Convert numpy array to QImage with correct color handling"""
        if isinstance(frame, QtGui.QImage):
            return frame

        if not isinstance(frame, np.ndarray):
            raise ValueError("Input must be numpy array")

        # 确保是3通道BGR格式
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]  # 去掉alpha通道
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 转换为RGB格式供QImage使用
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, _ = frame_rgb.shape
        bytes_per_line = 3 * width

        return QtGui.QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format_RGB888
        )

    def estimate_depth2(self, qimg, centers):
        try:
            # 1. 转换QImage为NumPy数组 (BGR格式)
            img_bgr = self.qimage2numpy2(qimg)
            # 2. 深度估计需要RGB格式
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # 3. 执行深度估计
            depth = self.depth_anything.infer_image(img_rgb, 518)
            if depth is None:
                print("深度估计失败")
                return
            # 4. 直接在原图上绘制中心点和深度值
            output_img = img_bgr.copy()  # 保留原始图像
            for center in centers:
                x, y = center
                if 0 <= x < output_img.shape[1] and 0 <= y < output_img.shape[0]:
                    # 绘制绿色中心点
                    cv2.circle(output_img, (x, y), 5, (0, 255, 0), -1)
                    # 获取深度值并显示
                    depth_value = depth[y, x]
                    cv2.putText(
                        output_img,
                        f"{depth_value:.2f}",
                        (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255),1)
            # 5. 显示处理后的图像
            qimg_result = self.convert_to_qimage(output_img)
            self.label.setPixmap(QtGui.QPixmap.fromImage(qimg_result))

        except Exception as e:
            print(f"深度处理错误: {str(e)}")
            import traceback
            traceback.print_exc()


    def estimate_depth(self, qimg):
        try:
            # 转换 QImage 为 NumPy 数组
            img_array = self.qimage2numpy(qimg)

            # 进行深度估计
            depth = self.depth_anything.infer_image(img_array, 518)
            if depth is None or depth.size == 0:
                print("深度估计失败或结果为空")
                return

            # 归一化深度图
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)

            # 颜色映射
            depth_colored = (self.cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            height, width, channel = depth_colored.shape
            bytes_per_line = channel * width
            qimg_depth = QtGui.QImage(depth_colored.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

            # 更新界面
            self.latest_image_processed = True
            self.label.setPixmap(QtGui.QPixmap.fromImage(qimg_depth))
        except Exception as e:
            print(f"深度估计过程中出现错误: {e}")

    def on_image_click(self, event):
        if event.button() == Qt.LeftButton:
            if self.double_click_timer.isActive():
                self.double_click_timer.stop()
                print("双击事件")
                self.current_mode = "point"
                self.mode_changed = True
            else:
                self.double_click_timer.start(500)  # 500 毫秒内算作双击

    def handle_single_click(self):
        print("单击事件")
        if self.current_mode == "normal":
            self.current_mode = "gray"
            self.gray_mode = True
        else:
            self.current_mode = "normal"
            self.gray_mode = False
        self.mode_changed = True

    def save_fire_image(self, qimg):
        """保存火灾检测画面到本地"""
        try:
            # 创建保存目录
            save_dir = r"C:\Users\19736\Desktop\FireDetections"
            os.makedirs(save_dir, exist_ok=True)

            # 生成带时间戳的文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fire_detection_{timestamp}.jpg"
            save_path = os.path.join(save_dir, filename)

            # 将QImage转换为QPixmap再保存
            pixmap = QtGui.QPixmap.fromImage(qimg)
            if not pixmap.isNull():
                success = pixmap.save(save_path, "JPG", quality=95)
                if success:
                    print(f"火灾画面已保存至: {save_path}")
                    return save_path
                else:
                    print("保存失败: QPixmap保存返回False")
            else:
                print("保存失败: QPixmap转换失败")
            return None
        except Exception as e:
            print(f"保存火灾画面失败: {str(e)}")
            return None


    # def update_classes(self, classes):
    #     current_classes = []
    #     if '0' in classes:
    #         self.detected_classes.add("烟")
    #         current_classes.append("烟")
    #         self.last_detection_time = QtCore.QDateTime.currentDateTime()
    #     if '1' in classes:
    #         self.detected_classes.add("火")
    #         current_classes.append("火")
    #         self.last_detection_time = QtCore.QDateTime.currentDateTime()
    #     self.textEdit_3.clear()
    #     self.textEdit_2.clear()
    #     if current_classes:
    #         self.textEdit_3.append("现检测到类别：" + ", ".join(current_classes) + "。")
    #         gif_path = "危险.gif"
    #     else:
    #         self.textEdit_3.append("未检测出火灾。")
    #         gif_path = "安全.gif"
    #     if self.gif_path != gif_path:
    #         self.gif_path = gif_path
    #         if self.movie is not None:
    #             self.movie.stop()
    #         self.movie = QtGui.QMovie(gif_path)
    #         self.gif_label.setMovie(self.movie)
    #         self.movie.start()

    def play_alarm_sound(self):
        """播放警报音"""
        try:
            self.player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile("alarm.wav")))
            self.player.play()
            self.is_alarm_playing = True
        except Exception as e:
            print(f"播放警报音失败: {e}")

    def stop_alarm_sound(self):
        """停止警报音"""
        self.player.stop()
        self.is_alarm_playing = False
    def update_classes(self, classes):
        current_classes = []
        fire_detected = False

        if '0' in classes:
            self.detected_classes.add("烟")
            current_classes.append("烟")
            self.last_detection_time = QtCore.QDateTime.currentDateTime()

        if '1' in classes:
            self.detected_classes.add("火")
            current_classes.append("火")
            fire_detected = True
            self.last_detection_time = QtCore.QDateTime.currentDateTime()

            # 保存火灾画面
            if hasattr(self, 'latest_image') and not self.latest_image.isNull():
                saved_path = self.save_fire_image(self.latest_image)
                if saved_path:
                    self.textEdit.append(f"火灾画面已保存: {saved_path}")
                    # 添加红色警示
                    #self.textEdit.append('<span style="color:red;">⚠️ 检测到火灾！请立即处理！</span>')
                    # 播放警报音
                    if not self.is_alarm_playing:
                        self.play_alarm_sound()

        self.textEdit_3.clear()
        self.textEdit_2.clear()

        if current_classes:
            status = "现检测到类别：" + ", ".join(current_classes) + "。"
            self.textEdit_3.append(status)
            gif_path = "危险.gif"
        else:
            status = "未检测出火灾。"
            self.textEdit_3.append(status)
            gif_path = "安全.gif"
            if self.is_alarm_playing:
                self.stop_alarm_sound()

        if self.gif_path != gif_path:
            self.gif_path = gif_path
            if self.movie is not None:
                self.movie.stop()
            self.movie = QtGui.QMovie(gif_path)
            self.gif_label.setMovie(self.movie)
            self.movie.start()
    def check_detection_status(self):
        current_time = QtCore.QDateTime.currentDateTime()
        if self.last_detection_time:
            time_diff = self.last_detection_time.secsTo(current_time)
            if time_diff<=10 and self.detected_classes:
                ten_seconds_ago = current_time.addSecs(-10)
                message = f"{ten_seconds_ago.toString('hh:mm:ss')} -{current_time.toString('hh:mm:ss')} 检测到: {', '.join(self.detected_classes)}。"
                self.detected_classes.clear()
                self.last_cleared_time = current_time
            else:
                message = f"{current_time.toString('hh:mm:ss')} 未检测出火灾。"
            self.textEdit.append(message)
            if self.textEdit.document().blockCount() > 10:
                cursor = self.textEdit.textCursor()
                cursor.movePosition(QtGui.QTextCursor.Start)
                cursor.movePosition(QtGui.QTextCursor.Down, QtGui.QTextCursor.KeepAnchor, 1)
                cursor.removeSelectedText()
                cursor.deleteChar()
    def update_time(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d   %H:%M:%S")
        self.time_label.setText(current_time)
    def update_status(self, current_classes):
        if current_classes:
            self.textEdit_2.setHtml("<img src='C:/path_to_your_image/fire_image.png' width='100%' height='100%'/>")
        else:
            self.textEdit_2.setHtml("<span style='color: green;'>一切正常</span>")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    login_window = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(login_window)
    login_window.show()
    sys.exit(app.exec())