import cv2
import numpy as np
import math
from ultralytics import YOLO
import csv
import os

# کلاس فیلتر کالمن چندمتغیره
class KalmanFilterMultivariate:
    def __init__(self, initial_state, transition_matrix, control_matrix, observation_matrix, process_covariance, measurement_covariance):
        self.state = initial_state
        self.transition_matrix = transition_matrix
        self.control_matrix = control_matrix
        self.observation_matrix = observation_matrix
        self.process_covariance = process_covariance
        self.measurement_covariance = measurement_covariance
        self.estimate_covariance = np.eye(len(initial_state))

    def predict(self, control_vector=np.zeros(1)):
        self.state = np.dot(self.transition_matrix, self.state) + np.dot(self.control_matrix, control_vector)
        self.estimate_covariance = np.dot(np.dot(self.transition_matrix, self.estimate_covariance), self.transition_matrix.T) + self.process_covariance
        return self.state

    def update(self, measurement):
        innovation = measurement - np.dot(self.observation_matrix, self.state)
        innovation_covariance = np.dot(np.dot(self.observation_matrix, self.estimate_covariance), self.observation_matrix.T) + self.measurement_covariance
        kalman_gain = np.dot(np.dot(self.estimate_covariance, self.observation_matrix.T), np.linalg.inv(innovation_covariance))
        self.state = self.state + np.dot(kalman_gain, innovation)
        self.estimate_covariance = self.estimate_covariance - np.dot(np.dot(kalman_gain, self.observation_matrix), self.estimate_covariance)
        return self.state

# بارگیری پارامترهای کالیبراسیون
datac = np.load('c:/Users/rezaa/PycharmProjects/pythonProject3/Amir96c.npz')
mtx_l = datac['camera_matrix_l']
dist_l = datac['dist_coeffs_l']
mtx_r = datac['camera_matrix_r']
dist_r = datac['dist_coeffs_r']
R = datac['R']
T = datac['T']

datap = np.load('c:/Users/rezaa/PycharmProjects/pythonProject3/Amir96p.npz')
R1 = datap['R1']
R2 = datap['R2']
P1 = datap['P1']
P2 = datap['P2']
Q = datap['Q']

# مقداردهی اولیه مدل YOLO
model = YOLO('c:/Users/rezaa/PycharmProjects/pythonProject3/ToyotaCameryHybrid')

# نام کلاس‌ها
classNames = ['C - M -B - L', 'C - M -G', 'C - M-W', 'C - N -B - L', 'C - N -G', 'C - N-W']

# رنگ‌ها برای جعبه‌های مرزی
class_colors = {
    "A20": (255, 0, 0),  # Blue
    "B10": (0, 255, 100),  # Green
    "C30": (0, 0, 255),  # Red
    "D80": (0, 0, 0),  # Black
    "E40": (255, 0, 255),  # Magenta
    "F60": (0, 255, 255),  # Yellow
    "G50": (0, 100, 100), # Dark green
}

# پارامترهای دوربین
camera_separation = 9.2  # فاصله بین دو دوربین (سانتی‌متر)

# فاصله واقعی (Ground Truth)
ground_truth_distance = float(input("Enter the ground truth distance (in cm) for the object: "))

# تابع محاسبه فاصله
def calculate_distance(disparity, focal_length, baseline):
    if disparity == 0:
        disparity = 0.001  # جلوگیری از تقسیم بر صفر
    distance = (focal_length * baseline) / disparity
    return distance

# تابع محاسبه زوایای اویلر
def calculate_euler_angles(x_left, y_left, x_right, y_right, width, height, mtx_l, mtx_r):
    cx_l, cy_l = mtx_l[0, 2], mtx_l[1, 2]
    cx_r, cy_r = mtx_r[0, 2], mtx_r[1, 2]

    yaw = math.atan2(x_left - cx_l, mtx_l[0, 0])
    pitch = math.atan2(y_left - cy_l, mtx_l[1, 1])
    roll = math.atan2(x_right - cx_r, mtx_r[0, 0]) - yaw

    yaw = math.degrees(yaw)
    pitch = math.degrees(pitch)
    roll = math.degrees(roll)

    return pitch, yaw, roll

def calculate_3d_coordinates(x_left, y_left, disparity, mtx_l, baseline):
    focal_length = mtx_l[0, 0]
    cx, cy = mtx_l[0, 2], mtx_l[1, 2]

    Z = (focal_length * baseline) / disparity
    X = (Z * (x_left - cx)) / focal_length
    Y = (Z * (y_left - cy)) / focal_length

    return X, Y, Z

# تابع رسم دایره مرکزی
def draw_center_circle(frame):
    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), 2)

# تابع رسم محورها
def draw_axes(frame):
    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2

    cv2.line(frame, (center_x, center_y), (center_x + 50, center_y), (0, 0, 255), 2)
    cv2.putText(frame, 'X', (center_x + 55, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.line(frame, (center_x, center_y), (center_x, center_y + 50), (255, 0, 0), 2)
    cv2.putText(frame, 'Y', (center_x, center_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# تابع تشخیص لبه‌ها با الگوریتم Canny
def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    return edges

# تابع تشخیص گوشه‌ها با الگوریتم Shi-Tomasi
def detect_corners(image, max_corners=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    corners = cv2.goodFeaturesToTrack(blurred, maxCorners=max_corners, qualityLevel=0.02, minDistance=15)
    corners = np.int0(corners)
    return corners

# مقداردهی اولیه دوربین‌ها
cap_left = cv2.VideoCapture(0)  # دوربین چپ
cap_right = cv2.VideoCapture(3)  # دوربین راست

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: Unable to open one or both cameras")
    exit()

# مسیر ذخیره فایل‌ها
output_path = r"C:\Users\rezaa\PycharmProjects\pythonProject3\1403.07.14"

# ایجاد فایل‌های CSV
kalman_file = os.path.join(output_path, "kalman_filter_results.csv")
with open(kalman_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Object ID", "X", "Y", "Distance (cm)", "Yaw (degrees)", "Predicted X", "Predicted Y", "Predicted Distance (cm)", "Predicted Yaw (degrees)"])

distance_file = os.path.join(output_path, "object_distances.csv")
with open(distance_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Object ID", "Class Label", "Distance (cm)"])

euler_angles_file = os.path.join(output_path, "euler_angles.csv")
with open(euler_angles_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Object ID", "Class Label", "Pitch (degrees)", "Yaw (degrees)", "Roll (degrees)"])

position_file = os.path.join(output_path, "object_positions.csv")
with open(position_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Object ID", "Class Label", "X", "Y", "Z"])

distance_error_file = os.path.join(output_path, "distance_errors.csv")
with open(distance_error_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Object ID", "Class Label", "Estimated Distance (cm)", "Ground Truth Distance (cm)", "Error (cm)"])

kalman_evaluation_file = os.path.join(output_path, "kalman_evaluation.csv")
with open(kalman_evaluation_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Object ID", "Raw X", "Raw Y", "Raw Distance", "Raw Yaw (degrees)", "Filtered X", "Filtered Y", "Filtered Distance", "Filtered Yaw (degrees)"])

# مقداردهی اولیه فیلتر کالمن (اضافه کردن Yaw به بردار حالت)
initial_state = np.array([0, 0, 0, 0])  # [X, Y, Distance, Yaw]
transition_matrix = np.eye(4)  # ماتریس انتقال حالت 4x4
control_matrix = np.zeros((4, 1))  # ماتریس کنترل 4x1
observation_matrix = np.eye(4)  # ماتریس مشاهده 4x4
process_covariance = np.eye(4) * 1e-4  # کوواریانس فرآیند 4x4
measurement_covariance = np.eye(4) * 1e-2  # کوواریانس اندازه‌گیری 4x4

kf_multivariate = KalmanFilterMultivariate(initial_state, transition_matrix, control_matrix, observation_matrix, process_covariance, measurement_covariance)

# لیست برای ذخیره داده‌های خام و فیلترشده
raw_data = []
filtered_data = []

# تابع پردازش فریم‌ها
def process_frames(img_left, img_right):
    h, w = img_left.shape[:2]
    map1x, map1y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (w, h), cv2.CV_32FC1)
    img_left_rect = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
    img_right_rect = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

    results_left = model(img_left_rect)
    results_right = model(img_right_rect)

    for r_left, r_right in zip(results_left, results_right):
        boxes_left = r_left.boxes
        boxes_right = r_right.boxes

        for boxL in boxes_left:
            clsID = int(boxL.cls[0])
            conf = boxL.conf[0]
            if conf < 0.5:
                continue
            xL, yL, wL, hL = boxL.xywh[0]

            for boxR in boxes_right:
                clsID_R = int(boxR.cls[0])
                conf_R = boxR.conf[0]
                if conf_R < 0.5:
                    continue
                xR, yR, wR, hR = boxR.xywh[0]

                if clsID == clsID_R:
                    label = classNames[clsID]

                    disparity = xL.cpu().item() - xR.cpu().item()
                    focal_length = (P1[0, 0] + P2[0, 0]) / 2
                    distance = calculate_distance(disparity, focal_length, camera_separation)
                    pitch, yaw, roll = calculate_euler_angles(xL.cpu().item(), yL.cpu().item(), xR.cpu().item(), yR.cpu().item(), wL.cpu().item(), hL.cpu().item(), mtx_l, mtx_r)

                    # محاسبه مختصات سه‌بعدی
                    X, Y, Z = calculate_3d_coordinates(xL.cpu().item(), yL.cpu().item(), disparity, mtx_l, camera_separation)

                    # پیش‌بینی و به‌روزرسانی فیلتر کالمن (اضافه کردن Yaw)
                    current_state = np.array([xL.cpu().item(), yL.cpu().item(), distance, yaw])
                    predicted_state = kf_multivariate.predict()
                    updated_state = kf_multivariate.update(current_state)

                    # محاسبه خطای فاصله‌سنجی
                    distance_error = abs(distance - ground_truth_distance)

                    # ذخیره داده‌های خام و فیلترشده برای ارزیابی کالمن
                    raw_data.append(current_state)
                    filtered_data.append(updated_state)

                    # رسم جعبه مرزی و اطلاعات
                    color = class_colors[label]
                    x1, y1 = int(xL.cpu().item() - wL.cpu().item()/2), int(yL.cpu().item() - hL.cpu().item()/2)
                    x2, y2 = int(xL.cpu().item() + wL.cpu().item()/2), int(yL.cpu().item() + hL.cpu().item()/2)
                    cv2.rectangle(img_left_rect, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_left_rect, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    cv2.putText(img_left_rect, f'Distance: {distance:.2f} cm', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    cv2.putText(img_left_rect, f'Error: {distance_error:.2f} cm', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    cv2.putText(img_left_rect, f'Pitch: {pitch:.2f} (x)', (x1, y1 - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    cv2.putText(img_left_rect, f'Yaw: {yaw:.2f} (y)', (x1, y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    cv2.putText(img_left_rect, f'Roll: {roll:.2f} (z)', (x1, y1 - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    cv2.putText(img_left_rect, f'3D Position: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}', (x1, y1 - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                    # تشخیص لبه‌ها و گوشه‌ها
                    object_roi = img_left_rect[y1:y2, x1:x2]
                    edges = detect_edges(object_roi)
                    corners = detect_corners(object_roi)

                    # نمایش لبه‌ها و گوشه‌ها
                    img_left_rect[y1:y2, x1:x2][edges != 0] = [0, 255, 0]
                    for corner in corners:
                        x_corner, y_corner = corner.ravel()
                        cv2.circle(img_left_rect, (x1 + x_corner, y1 + y_corner), 3, (255, 0, 0), -1)

                    # ذخیره اطلاعات در فایل‌های CSV
                    with open(kalman_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([clsID, xL.cpu().item(), yL.cpu().item(), distance, yaw, predicted_state[0], predicted_state[1], predicted_state[2], predicted_state[3]])

                    with open(distance_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([clsID, label, distance])

                    with open(distance_error_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([clsID, label, distance, ground_truth_distance, distance_error])

                    with open(euler_angles_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([clsID, label, pitch, yaw, roll])

                    with open(position_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([clsID, label, X, Y, Z])

                    with open(kalman_evaluation_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([clsID, current_state[0], current_state[1], current_state[2], current_state[3], updated_state[0], updated_state[1], updated_state[2], updated_state[3]])

    return img_left_rect, img_right_rect

# حلقه اصلی
while cap_left.isOpened() and cap_right.isOpened():
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Error: Unable to read frames from one or both cameras")
        break

    frame_left_rect, frame_right_rect = process_frames(frame_left, frame_right)

    # رسم دایره مرکزی و محورها
    draw_center_circle(frame_left_rect)
    draw_axes(frame_left_rect)
    draw_center_circle(frame_right_rect)
    draw_axes(frame_right_rect)

    # نمایش فریم‌ها
    cv2.imshow("Left Camera", frame_left_rect)
    cv2.imshow("Right Camera", frame_right_rect)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# محاسبه معیارهای ارزیابی فیلتر کالمن
raw_data = np.array(raw_data)
filtered_data = np.array(filtered_data)

if len(raw_data) > 0 and len(filtered_data) > 0:
    # محاسبه MSE برای X
    mse_x = np.mean((raw_data[:, 0] - filtered_data[:, 0])**2)
    # محاسبه MSE برای Y
    mse_y = np.mean((raw_data[:, 1] - filtered_data[:, 1])**2)
    # محاسبه MSE برای Distance
    mse_distance = np.mean((raw_data[:, 2] - filtered_data[:, 2])**2)
    # محاسبه MSE برای Yaw
    mse_yaw = np.mean((raw_data[:, 3] - filtered_data[:, 3])**2)

    # محاسبه واریانس داده‌های خام و فیلترشده
    variance_raw_x = np.var(raw_data[:, 0])
    variance_filtered_x = np.var(filtered_data[:, 0])
    noise_reduction_x = ((variance_raw_x - variance_filtered_x) / variance_raw_x) * 100 if variance_raw_x != 0 else 0

    variance_raw_y = np.var(raw_data[:, 1])
    variance_filtered_y = np.var(filtered_data[:, 1])
    noise_reduction_y = ((variance_raw_y - variance_filtered_y) / variance_raw_y) * 100 if variance_raw_y != 0 else 0

    variance_raw_distance = np.var(raw_data[:, 2])
    variance_filtered_distance = np.var(filtered_data[:, 2])
    noise_reduction_distance = ((variance_raw_distance - variance_filtered_distance) / variance_raw_distance) * 100 if variance_raw_distance != 0 else 0

    variance_raw_yaw = np.var(raw_data[:, 3])
    variance_filtered_yaw = np.var(filtered_data[:, 3])
    noise_reduction_yaw = ((variance_raw_yaw - variance_filtered_yaw) / variance_raw_yaw) * 100 if variance_raw_yaw != 0 else 0

    # ذخیره معیارهای ارزیابی
    kalman_metrics_file = os.path.join(output_path, "kalman_metrics.csv")
    with open(kalman_metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["MSE X", mse_x])
        writer.writerow(["MSE Y", mse_y])
        writer.writerow(["MSE Distance", mse_distance])
        writer.writerow(["MSE Yaw", mse_yaw])
        writer.writerow(["Noise Reduction X (%)", noise_reduction_x])
        writer.writerow(["Noise Reduction Y (%)", noise_reduction_y])
        writer.writerow(["Noise Reduction Distance (%)", noise_reduction_distance])
        writer.writerow(["Noise Reduction Yaw (%)", noise_reduction_yaw])
        writer.writerow(["Variance Raw Yaw (degrees^2)", variance_raw_yaw])
        writer.writerow(["Variance Filtered Yaw (degrees^2)", variance_filtered_yaw])

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()