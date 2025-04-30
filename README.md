Automated Connector Detection for Toyota Camry Hybrid Battery
Overview
This project implements an automated connector detection system for Toyota Camry Hybrid battery connectors using YOLOv11. The system leverages stereo vision to estimate the position and orientation of connectors, enhanced by a Kalman filter for noise reduction. The project includes training scripts, pre-trained models, sample datasets, and detailed performance analysis.
Features

Connector Detection: Detects 19 different connector types using YOLOv11.
Stereo Vision: Uses left and right cameras to estimate connector position (X, Y, Z) and orientation (pitch, yaw, roll).
Kalman Filter: Reduces noise in distance measurements for more accurate pose estimation.
Performance Evaluation: Includes detailed metrics (mAP, FPS) and comparisons with other models (Mask R-CNN, SSD, Faster R-CNN).

Requirements
To run this project, you'll need the following:

Python 3.10+
Ultralytics 8.3.53
PyTorch 2.5.1
Roboflow (for dataset access)

Install the dependencies:
pip install ultralytics torch roboflow

Dataset
The dataset is sourced from Roboflow and includes:

571 training images
79 validation images
19 connector classes (e.g., A20, B10, C30, etc.)

A sample subset of the dataset (50 images with labels) is available at /data. The dataset configuration file (dataset.yaml) is also provided in the same directory.
Class Distribution: Some classes (e.g., C30) have ~100 instances, while others (e.g., 16PinBlackW) have only 2 instances, leading to performance variations.
Setup and Usage
1. Clone the Repository
git clone https://github.com/amirreza986/Automated-connector-detection-.git
cd Automated-connector-detection-

2. Train the Model
To train the YOLOv11 model on your dataset:
python scripts/Detail.png

This script uses the configuration in /data/dataset.yaml and saves the trained model to /models.
3. Run Inference
To detect connectors on new images or videos using the pre-trained model:
python scripts/detect_connectors.py --source path/to/image_or_video --model models/best.pt

Results will be saved in the runs/detect directory.
4. Test the Model
To test the model on sample images:
python scripts/test_model.py

This script runs inference on images in /data/test/images and saves the results.
Pre-trained Model
A pre-trained YOLOv11 model is available at /models/best.pt. You can use it directly for inference without retraining.
Results
Performance Metrics
After training for 200 epochs on a Tesla T4 GPU:

mAP@0.5: 0.408
mAP@0.5:0.95: 0.319
Inference Speed: 26 ms per image (suitable for real-time applications)

Best performance was observed for class B10 (mAP@0.5 = 0.934), while classes with fewer samples (e.g., 16PinBlackW, 20PinBlackW) had mAP@0.5 = 0 due to data imbalance.
Sample Results
Below are sample detections from the left and right cameras, showing the system's ability to detect connectors and provide positional data:
 Camera DetectionThis image-1 shows the detection of a connector using the left camera, with positional data including offset (-52.01 mm, -11.96 mm), pitch (-1.56°), yaw (1.51°), and distance (37.96 cm).
 Camera DetectionThis image-2 shows the detection of a connector using the right camera, with positional data including offset (107.53 mm, 15.57 mm), pitch (10.30°), yaw (0.00°), and distance (37.71 cm).
Performance Analysis
Error Analysis
The following boxplot shows the error in distance estimation at different depths, with the ideal error threshold marked in red:

Kalman Filter Performance
Raw Distance MeasurementsThis plot shows the raw distance measurements before applying the Kalman filter at different depths:

Filtered Distance MeasurementsThis plot shows the filtered distance measurements after applying the Kalman filter, demonstrating reduced noise:

Pose Estimation Analysis
The following plots compare the pitch, yaw, roll, and coordinates (X, Y, Z) at different depths:
Pitch Comparison
Yaw Comparison
Roll Comparison
X Coordinates Comparison
Y Coordinates Comparison
Z Coordinates Comparison
Model Comparison
The following chart compares the accuracy and frames per second (FPS) of different models:

YOLOv11 achieves the highest accuracy (92.5%) while maintaining a high FPS (50), making it the best choice for real-time connector detection.
Limitations and Future Work

Limitations: The model struggles with classes having few samples (e.g., 16PinBlackW, 20PinBlackW) due to imbalanced data.
Future Work: Collect more data for underrepresented classes, apply advanced data augmentation, or explore transfer learning with larger datasets.

Additional Documentation

Data Collection Details
Training Details
Performance Analysis
Pose Estimation Analysis
Model Comparison

License
This project is licensed under the MIT License - see the LICENSE file for details.
