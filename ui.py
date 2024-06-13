import sys
from PyQt5.QtWidgets import (QMainWindow, QApplication, QFileDialog, QLabel, QMessageBox, QAction, QVBoxLayout, QWidget)
from PyQt5.QtGui import QPixmap, QPalette, QColor
from PyQt5.QtCore import Qt
import torch
from torchvision import transforms
from PIL import Image
import os
import cv2  # Ensure you import cv2
from utils.segment import split_image
from mnist_classify import load_model
from PyQt5.QtGui import QFont

def predict(model, device, image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    pred = output.max(1, keepdim=True)[1]
    return pred.item()

def predict_on_segments(segment_paths, model, device):
    results = []
    for segment_path in segment_paths:
        result = predict(model, device, segment_path)
        results.append(result)
    return results

def predictions_to_string(predictions):
    return ''.join(map(str, predictions))

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.path = ''
        self.cwd = os.getcwd()  # 当前工作目录
        self.change_path = "change.png"  # 被处理过的图像的路径
        self.initUI()

    def initUI(self):
        self.setWindowTitle('MNIST 识别系统')
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.lbl = QLabel(self)
        self.lbl.setAlignment(Qt.AlignCenter)  # 图像显示区，居中
        self.lbl.setStyleSheet("border: 2px solid black")
        layout.addWidget(self.lbl)

        self.label1 = QLabel("识别结果", self)
        self.label1.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label1)

        self.createMenu()
        self.setGeometry(50, 50, 600, 600)  # 设置初始窗口大小
        self.show()

    def createMenu(self):
        menubar = self.menuBar()
        menu1 = menubar.addMenu("文件")
        open_action = QAction("打开", self)
        menu1.addAction(open_action)
        open_action.triggered.connect(self.menu1_process)

    def menu1_process(self):
        self.path, _ = QFileDialog.getOpenFileName(self, '打开文件', self.cwd,
                                                   "All Files (*);;Images (*.bmp *.tif *.png *.jpg)")
        print(f"Selected file path: {self.path}")
        if not self.path:
            QMessageBox.warning(self, "错误", "未选择文件或文件无效")
            return

        self.image = cv2.imread(self.path)
        if self.image is None:
            QMessageBox.warning(self, "错误", "无法读取图像文件")
            return

        # 保留原图像用于分割和预测
        cv2.imwrite(self.change_path, self.image)
        
        # 创建放大版本的图像用于显示
        display_image = cv2.resize(self.image, (224, 224))  # 调整到合适的尺寸
        display_image_path = "display_image.png"
        cv2.imwrite(display_image_path, display_image)
        
        # 在 QLabel 上显示放大后的图像
        self.lbl.setPixmap(QPixmap(display_image_path))
        self.lbl.setScaledContents(True)  # 确保 QLabel 会缩放图像以适应
        self.adjustSize()

        # 处理图像并预测
        segment_image_paths = split_image(self.change_path, 32, 32)
        predictions = predict_on_segments(segment_image_paths, model, DEVICE)
        predictions_str = predictions_to_string(predictions)

        # 在界面上显示预测结果
        self.label1.setText(f"识别结果: {predictions_str}")
        self.label1.adjustSize()  # 调整标签大小以适应文本
        font = QFont()
        font.setPointSize(16)  # 设置字体大小，例如16
        self.label1.setFont(font)

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './models/mnist_cnn.pt'
    model = load_model(model_path, DEVICE)

    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())
