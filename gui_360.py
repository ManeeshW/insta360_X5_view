import sys
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QCheckBox, QSlider, QLineEdit, QFormLayout, QPushButton
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Insta360 Viewer")
        self.central = QWidget()
        self.setCentralWidget(self.central)
        layout = QHBoxLayout()
        self.central.setLayout(layout)

        # Left image display
        self.image_label = QLabel()
        self.image_label.setFixedSize(1280, 960)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self.image_label)

        # Right controls
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        layout.addWidget(right_widget)

        # Scale checkbox
        self.scale_check = QCheckBox("Scale to fit")
        self.scale_check.setChecked(False)
        self.scale_check.stateChanged.connect(self.on_scale_changed)
        right_layout.addWidget(self.scale_check)

        # Resolution controls
        res_layout = QFormLayout()
        self.res_w_edit = QLineEdit("640")
        self.res_w_edit.setFixedWidth(60)
        self.res_w_edit.returnPressed.connect(self.update_resolution)
        self.res_h_edit = QLineEdit("480")
        self.res_h_edit.setFixedWidth(60)
        self.res_h_edit.returnPressed.connect(self.update_resolution)
        res_layout.addRow("Width:", self.res_w_edit)
        res_layout.addRow("Height:", self.res_h_edit)
        right_layout.addLayout(res_layout)

        # FOV controls
        fov_layout = QFormLayout()
        self.fov_slider = QSlider(Qt.Orientation.Horizontal)
        self.fov_slider.setRange(10, 1790)  # 1.0 to 179.0, multiplied by 10
        self.fov_slider.setValue(825)
        self.fov_slider.valueChanged.connect(self.update_fov_edit)
        self.fov_edit = QLineEdit("82.5")
        self.fov_edit.setFixedWidth(60)
        self.fov_edit.returnPressed.connect(self.update_fov_slider)
        fov_layout.addRow("FOV:", self.fov_slider)
        fov_layout.addRow("", self.fov_edit)
        right_layout.addLayout(fov_layout)

        # Angle controls
        yaw_layout = QHBoxLayout()
        yaw_minus = QPushButton("-")
        yaw_minus.pressed.connect(lambda: self.set_yaw_dir(-1))
        yaw_minus.released.connect(lambda: self.set_yaw_dir(0))
        self.yaw_label = QLabel("Yaw: 0.0")
        yaw_plus = QPushButton("+")
        yaw_plus.pressed.connect(lambda: self.set_yaw_dir(1))
        yaw_plus.released.connect(lambda: self.set_yaw_dir(0))
        yaw_layout.addWidget(yaw_minus)
        yaw_layout.addWidget(self.yaw_label)
        yaw_layout.addWidget(yaw_plus)
        right_layout.addLayout(yaw_layout)

        pitch_layout = QHBoxLayout()
        pitch_minus = QPushButton("-")
        pitch_minus.pressed.connect(lambda: self.set_pitch_dir(-1))
        pitch_minus.released.connect(lambda: self.set_pitch_dir(0))
        self.pitch_label = QLabel("Pitch: 0.0")
        pitch_plus = QPushButton("+")
        pitch_plus.pressed.connect(lambda: self.set_pitch_dir(1))
        pitch_plus.released.connect(lambda: self.set_pitch_dir(0))
        pitch_layout.addWidget(pitch_minus)
        pitch_layout.addWidget(self.pitch_label)
        pitch_layout.addWidget(pitch_plus)
        right_layout.addLayout(pitch_layout)

        roll_layout = QHBoxLayout()
        roll_minus = QPushButton("-")
        roll_minus.pressed.connect(lambda: self.set_roll_dir(-1))
        roll_minus.released.connect(lambda: self.set_roll_dir(0))
        self.roll_label = QLabel("Roll: 0.0")
        roll_plus = QPushButton("+")
        roll_plus.pressed.connect(lambda: self.set_roll_dir(1))
        roll_plus.released.connect(lambda: self.set_roll_dir(0))
        roll_layout.addWidget(roll_minus)
        roll_layout.addWidget(self.roll_label)
        roll_layout.addWidget(roll_plus)
        right_layout.addLayout(roll_layout)

        # Reset button
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_orientation)
        right_layout.addWidget(reset_button)

        # Initialize angles and resolution
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.out_w = 640
        self.out_h = 480

        # Directions for continuous rotation
        self.yaw_dir = 0
        self.pitch_dir = 0
        self.roll_dir = 0
        self.rotation_speed = 90.0  # degrees per second

        # Time for delta calculation
        self.last_time = time.time()

        # Camera capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            print("Error: Could not open Insta360 camera.")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2880)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
            self.cap.set(cv2.CAP_PROP_FPS, 60)

        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)  # Approx 60 FPS

        # Set initial focus to image label
        self.image_label.setFocus()

    def set_yaw_dir(self, direction):
        self.yaw_dir = direction

    def set_pitch_dir(self, direction):
        self.pitch_dir = direction

    def set_roll_dir(self, direction):
        self.roll_dir = direction

    def update_labels(self):
        self.yaw_label.setText(f"Yaw: {self.yaw:.1f}")
        self.pitch_label.setText(f"Pitch: {self.pitch:.1f}")
        self.roll_label.setText(f"Roll: {self.roll:.1f}")

    def reset_orientation(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.update_labels()

    def update_fov_edit(self, value):
        fov = value / 10.0
        self.fov_edit.setText(f"{fov:.1f}")

    def update_fov_slider(self):
        try:
            fov = float(self.fov_edit.text())
            if 1 <= fov <= 179:
                self.fov_slider.setValue(int(fov * 10))
        except ValueError:
            pass
        self.image_label.setFocus()

    def update_resolution(self):
        try:
            w = int(self.res_w_edit.text())
            h = int(self.res_h_edit.text())
            if w > 0 and h > 0:
                self.out_w = w
                self.out_h = h
        except ValueError:
            pass
        self.image_label.setFocus()

    def on_scale_changed(self, state):
        pass  # No need for action here, checked in update_frame

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        key = event.text()
        if key == 'p':  # Pitch up
            self.set_pitch_dir(1)
        elif key == ';':  # Pitch down
            self.set_pitch_dir(-1)
        elif key == 'l':  # Yaw left
            self.set_yaw_dir(-1)
        elif key == "'":  # Yaw right
            self.set_yaw_dir(1)
        elif key == 'o':  # Roll left
            self.set_roll_dir(-1)
        elif key == '[':  # Roll right
            self.set_roll_dir(1)
        elif key in ['/', '.']:
            self.reset_orientation()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        key = event.text()
        if key in ['p', ';']:
            self.set_pitch_dir(0)
        elif key in ['l', "'"]:
            self.set_yaw_dir(0)
        elif key in ['o', '[']:
            self.set_roll_dir(0)

    def update_frame(self):
        # Update angles based on directions
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.yaw += self.yaw_dir * self.rotation_speed * dt
        self.pitch += self.pitch_dir * self.rotation_speed * dt
        self.roll += self.roll_dir * self.rotation_speed * dt
        self.update_labels()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        fov = self.fov_slider.value() / 10.0
        perspective = self.equi_to_persp(frame, self.yaw, self.pitch, self.roll, fov, self.out_w, self.out_h)

        # Prepare display image
        if self.scale_check.isChecked():
            display_img = cv2.resize(perspective, (1280, 960), interpolation=cv2.INTER_LINEAR)
        else:
            display_img = np.zeros((960, 1280, 3), dtype=np.uint8)
            start_y = (960 - self.out_h) // 2
            start_x = (1280 - self.out_w) // 2
            end_y = start_y + self.out_h
            end_x = start_x + self.out_w
            # Handle if output larger than display
            if start_y < 0 or start_x < 0 or end_y > 960 or end_x > 1280:
                # Crop to fit
                crop_start_y = max(0, -start_y)
                crop_start_x = max(0, -start_x)
                crop_end_y = min(self.out_h, self.out_h - (end_y - 960))
                crop_end_x = min(self.out_w, self.out_w - (end_x - 1280))
                display_img[0:960, 0:1280] = perspective[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
            else:
                display_img[start_y:end_y, start_x:end_x] = perspective

        # Convert to RGB for Qt
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        qimg = QImage(display_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def equi_to_persp(self, equi_img, yaw, pitch, roll, fov, out_w, out_h):
        fov_rad = np.deg2rad(fov)
        f = out_w / 2.0 / np.tan(fov_rad / 2)

        xx, yy = np.meshgrid(np.arange(out_w), np.arange(out_h))
        cam_x = ((xx - out_w / 2.0) / f).astype(np.float64)
        cam_y = ((out_h / 2.0 - yy) / f).astype(np.float64)
        cam_z = np.ones_like(xx, dtype=np.float64)

        dist = np.sqrt(cam_x**2 + cam_y**2 + cam_z**2)
        cam_x /= dist
        cam_y /= dist
        cam_z /= dist

        # Rotation matrices
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)
        roll_rad = np.deg2rad(roll)

        R_yaw = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        R_roll = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad), np.cos(roll_rad), 0],
            [0, 0, 1]
        ])

        # Combined rotation: yaw * pitch * roll
        R = np.dot(np.dot(R_yaw, R_pitch), R_roll)

        # Apply rotation
        world_x = R[0, 0] * cam_x + R[0, 1] * cam_y + R[0, 2] * cam_z
        world_y = R[1, 0] * cam_x + R[1, 1] * cam_y + R[1, 2] * cam_z
        world_z = R[2, 0] * cam_x + R[2, 1] * cam_y + R[2, 2] * cam_z

        # Spherical coordinates
        theta = np.arctan2(world_x, world_z)
        phi = np.arcsin(world_y)

        # Map to equirectangular
        in_w = equi_img.shape[1]
        in_h = equi_img.shape[0]
        map_x = (theta / (2 * np.pi) + 0.5) * in_w
        map_y = (0.5 - phi / np.pi) * in_h

        # Remap
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        persp = cv2.remap(equi_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return persp

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())