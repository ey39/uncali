import sys
import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QTextEdit, QMessageBox, QSplitter)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import pickle
import time
from multiprocessing import shared_memory
import threading

class CoordinateVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.frames = {}  # 存储所有坐标系
        self.grid_item = None  # 存储网格项
        self.bg_dark = True  # 背景色状态
        self.showtext = False
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('实时坐标系可视化器')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 创建3D视图
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=5, elevation=30, azimuth=45)
        # 设置背景色为深灰色，更容易看见网格和坐标系
        self.gl_widget.setBackgroundColor(0.1, 0.1, 0.1, 1.0)  # 深灰色背景
        layout.addWidget(self.gl_widget, 1)
        
        # 创建控制按钮区域
        button_widget = QWidget()
        button_widget.setMaximumHeight(60)
        button_widget.setStyleSheet("background-color: #34495e; border-top: 1px solid #2c3e50;")
        
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(10, 10, 10, 10)
        
        qss_btn = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """

        # 重置视角按钮
        self.reset_button = QPushButton('重置视角')
        self.reset_button.clicked.connect(self.reset_view)
        self.reset_button.setStyleSheet(qss_btn)
        button_layout.addWidget(self.reset_button)
        
        # 清除按钮
        self.clear_button = QPushButton('清除所有坐标系')
        self.clear_button.clicked.connect(self.clear_all_frames)
        self.clear_button.setStyleSheet(qss_btn)
        button_layout.addWidget(self.clear_button)
        
        # 显示网格按钮
        self.toggle_grid_button = QPushButton('显示/隐藏网格')
        self.toggle_grid_button.clicked.connect(self.toggle_grid)
        self.toggle_grid_button.setStyleSheet(qss_btn)
        # 添加背景色切换按钮
        self.toggle_bg_button = QPushButton('切换背景')
        self.toggle_bg_button.clicked.connect(self.toggle_background)
        self.toggle_bg_button.setStyleSheet(qss_btn)
        button_layout.addWidget(self.toggle_bg_button)

        # 显示名称
        self.show_name_button = QPushButton('显示名称')
        self.show_name_button.clicked.connect(self.show_name)
        self.show_name_button.setStyleSheet(qss_btn)
        button_layout.addWidget(self.show_name_button)
        
        # 信息标签
        # self.info_label = QLabel('坐标系数量: 0 | 网格: 显示')
        # self.info_label.setStyleSheet("color: white; font-weight: bold;")
        # button_layout.addWidget(self.info_label)
        
        button_layout.addStretch()
        
        layout.addWidget(button_widget, 0)
        central_widget.setLayout(layout)
        
        # 添加网格
        self.add_grid()
        
        # 更新信息
        # self.update_info()
        
    def add_grid(self):
        """添加网格"""
        if self.grid_item is None:
            self.grid_item = gl.GLGridItem()
            self.grid_item.setSize(x=10, y=10, z=10)
            self.grid_item.setSpacing(x=0.5, y=0.5, z=0.5)
            # 使用更明亮的网格颜色
            self.grid_item.setColor('gray')
            self.gl_widget.addItem(self.grid_item)
    
    def toggle_background(self):
        """切换背景色"""
        if self.bg_dark:
            # 切换到浅色背景
            self.gl_widget.setBackgroundColor('gray')  # 浅灰色
            # 调整网格颜色以适应浅色背景
            if self.grid_item:
                self.grid_item.setColor('k')
            self.bg_dark = False
        else:
            # 切换到深色背景
            self.gl_widget.setBackgroundColor('k')  # 深灰色
            # 调整网格颜色以适应深色背景
            if self.grid_item:
                self.grid_item.setColor('gray')
            self.bg_dark = True
    
    def toggle_grid(self):
        """切换网格显示/隐藏"""
        if self.grid_item is not None:
            if self.grid_item.visible():
                self.grid_item.setVisible(False)
            else:
                self.grid_item.setVisible(True)
            self.update_info()
    
    def show_name(self):
        """切换坐标系名称显示/隐藏"""
        self.showtext = not self.showtext

    def add_coordinate_frame(self, name, position, rotation_matrix, scale=0.1):
        """添加坐标系
        
        Args:
            name: 坐标系名称
            position: 位置 [x, y, z]
            rotation_matrix: 3x3旋转矩阵
            scale: 坐标轴长度缩放
        """
        if name in self.frames:
            self.remove_coordinate_frame(name)
        
        frame_items = []
        
        # 坐标轴颜色 (红色X轴，绿色Y轴，蓝色Z轴)
        colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        axis_labels = ['X', 'Y', 'Z']
        
        # 创建坐标轴
        for i in range(3):
            # 计算轴的端点
            axis_end = np.array(position) + rotation_matrix[:, i] * scale
            
            # 创建轴线
            axis_line = gl.GLLinePlotItem(
                pos=np.array([position, axis_end]),
                color=colors[i],
                width=1
            )
            self.gl_widget.addItem(axis_line)
            frame_items.append(axis_line)
            if self.showtext:
                axis_text = gl.GLTextItem()
                axis_text.setData(
                    pos=axis_end, 
                    text=axis_labels[i], 
                    color='white',
                )
                frame_items.append(axis_text)
                self.gl_widget.addItem(axis_text)
        
        # 添加原点标记
        # origin_marker = gl.GLScatterPlotItem(
        #     pos=np.array(position),
        #     color=(1, 1, 1, 1),  # 白色
        #     size=15
        # )
        # frame_items.append(origin_marker)
        # self.gl_widget.addItem(origin_marker)
        if self.showtext:
            frame_text = gl.GLTextItem()
            frame_text.setData(
                pos=np.array(position)+np.array([0.0, 0.0, -0.1]), 
                text=name, 
                color='white',
            )
            frame_items.append(frame_text)
            self.gl_widget.addItem(frame_text)

        # 存储坐标系
        self.frames[name] = {
            'items': frame_items,
            'position': position,
            'rotation_matrix': rotation_matrix,
            'scale': scale
        }
        # self.update_info()

    
    def add_coordinate_frame_T(self, name, homogeneous_matrix, scale=1.0):
        rotation_matrix = homogeneous_matrix[:3, :3]
        position = homogeneous_matrix[:3, 3]
        # print("t:",position)
        # print("R:",rotation_matrix)
        self.add_coordinate_frame(name, position, rotation_matrix)

    
    def remove_coordinate_frame(self, name):
        """移除指定的坐标系"""
        if name in self.frames:
            for item in self.frames[name]['items']:
                self.gl_widget.removeItem(item)
            del self.frames[name]
            # self.update_info()
    
    def clear_all_frames(self):
        """清除所有坐标系"""
        for name in list(self.frames.keys()):
            self.remove_coordinate_frame(name)
    
    def reset_view(self):
        """重置视角"""
        self.gl_widget.setCameraPosition(distance=5, elevation=30, azimuth=45)
    
    # def update_info(self):
    #     """更新信息标签"""
    #     grid_status = "显示" if (self.grid_item and self.grid_item.visible()) else "隐藏"
    #     self.info_label.setText(f'坐标系数量: {len(self.frames)} | 网格: {grid_status}')
    
    def update_coordinate_frame(self, name, position=None, rotation_matrix=None, scale=None):
        """更新现有坐标系"""
        if name in self.frames:
            current_frame = self.frames[name]
            
            # 使用新值或保持原值
            new_position = position if position is not None else current_frame['position']
            new_rotation = rotation_matrix if rotation_matrix is not None else current_frame['rotation_matrix']
            new_scale = scale if scale is not None else current_frame['scale']
            
            # 重新添加坐标系
            self.add_coordinate_frame(name, new_position, new_rotation, new_scale)
    
    def get_frame_info(self, name):
        """获取坐标系信息"""
        if name in self.frames:
            return self.frames[name].copy()
        return None
    
    def list_frames(self):
        """列出所有坐标系名称"""
        return list(self.frames.keys())

def receive(visualizer):
    shm = None
    while shm is None:
        try:
            shm = shared_memory.SharedMemory(name="dict_shm")
        except FileNotFoundError:
            time.sleep(0.1)
    print("连接共享内存，开始读取数据")
    while True:
        size = int.from_bytes(shm.buf[:4], byteorder='little')
        data_bytes = shm.buf[4:4+size]
        data_dict = pickle.loads(data_bytes)
        for name, matrix in data_dict.items():
            # print(f"坐标系 [{name}] 的矩阵为：\n{matrix}")
            visualizer.add_coordinate_frame_T(name, matrix)
        time.sleep(0.1)

# 使用示例
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 创建可视化器
    visualizer = CoordinateVisualizer()
    visualizer.show()
    
    # 添加一些测试坐标系
    # 世界坐标系
    # visualizer.add_coordinate_frame(
    #     "world", 
    #     [0, 0, 0], 
    #     np.eye(3), 
    #     0.2
    # )
    
    t = threading.Thread(target=receive, args=(visualizer,))  
    t.start()
    sys.exit(app.exec_())
    