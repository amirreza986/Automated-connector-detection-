Automated Connector Detection and 3D Localization for EV Battery Recycling
This repository contains the code, datasets, models, and results for the paper "Automated connector detection and 3D localization for EV battery recycling using deep learning and Kalman filtering" published at ICMR 2025. The system integrates YOLOv11, stereo vision, and a multivariate Kalman filter to enable real-time connector detection and precise 3D localization for robotic disassembly of electric vehicle (EV) battery packs, tested on a 2016 Toyota Camry hybrid battery.
Overview
The proposed system addresses the challenges of manual EV battery disassembly by automating connector detection and 3D localization. Key features include:

YOLOv11-based Detection: Achieves 95.0% mAP@0.5 for connector detection (see validation results in /results/performance).
Stereo Vision: Provides sub-centimeter depth accuracy (median errors of 0.15-0.25 cm at 15.5-17.7 cm depths).
Kalman Filtering: Reduces yaw noise by 89.66% for stable robotic alignment.

The system enhances safety and scalability, contributing to sustainable EV battery recycling.
Repository Structure

/data: Sample dataset of battery connector images with annotations (50 images from the 500-image dataset).
/models: Pre-trained YOLOv11 model weights (best.pt).
/scripts: Python scripts for training, inference, stereo processing, and Kalman filtering (to be added).
/results:
/results/performance: Validation results table, confusion matrix, F1, precision, recall, and precision-recall curves.
/results/sample_detections: Sample detection images showcasing the model’s performance.
/results/training_metrics: Training loss and metrics plots (box, cls, dfl, precision, recall, mAP).
/results/dataset_analysis: Distribution of bounding boxes and class instances.


/configs: Training configuration file (args.yaml).
/docs: Experimental details and supplementary reports (to be added).
/videos: Demonstration videos of the system in action (to be added).

Installation

Clone the repository:git clone https://github.com/amirreza986/Automated-connector-detection-.git
cd Automated-connector-detection-


Install dependencies:pip install -r requirements.txt

Requirements include torch==2.0.1, opencv-python==4.7.0.72, scikit-learn==1.0.2, and torchvision==0.15.2.
Download the pre-trained model from /models.

Usage
1. Connector Detection
Run YOLOv11 inference on a sample image:
python scripts/detect_connectors.py --image data/sample_image.jpg --model models/best.pt

Output: Bounding boxes and class scores saved in /results/sample_detections.
2. 3D Localization
Compute 3D coordinates using stereo vision:
python scripts/stereo_localization.py --left data/left_image.jpg --right data/right_image.jpg --calibration scripts/calibration.yaml

Output: 3D coordinates (X, Y, Z) and depth maps in /results/localization.
3. Orientation Stabilization
Apply the Kalman filter to smooth orientation estimates:
python scripts/kalman_filter.py --input results/localization/poses.csv

Output: Filtered yaw angles in /results/filtered_poses.
Results

Validation Performance: Detailed metrics including precision, recall, and mAP for each class are available in /results/performance/validation_results_table.png.
Confusion Matrix: Normalized confusion matrix showing classification performance in /results/performance/confusion_matrix.png.
Performance Curves: F1, precision, recall, and precision-recall curves in /results/performance/curves.
Sample Detections: Images of detected connectors with bounding boxes and confidence scores in /results/sample_detections.
Training Metrics: Loss and performance metrics over 200 epochs in /results/training_metrics.
Dataset Analysis: Distribution of bounding boxes and class instances in /results/dataset_analysis.

Dataset
The /data folder contains a sample of 50 labeled images from the 500-image dataset used in the paper. Annotations are provided in YOLO format (.txt files). To train the model on your own data, update the paths in scripts/train_yolov11.py.
Training
To train the YOLOv11 model using the provided configuration:
python scripts/train_yolov11.py --data data/dataset.yaml --epochs 200 --batch-size 4 --cfg configs/args.yaml

Pre-trained weights are available in /models for immediate use. Training metrics are available in /results/training_metrics.
Demonstration
Watch a video of the system detecting and localizing connectors in real-time at /videos/demo.mp4 (to be added).
Citation
If you use this work, please cite:
@article{khanloo2025automated,
  title={Automated connector detection and 3D localization for EV battery recycling using deep learning and Kalman filtering},
  author={Khanloo, Amirreza and Sorouri, Majid and Lacey, Gerrard},
  journal={ICMR},
  year={2025}
}

Contact
For questions, contact Amirreza Khanloo (amirreza986@gmail.com) or Majid Sorouri (majid.sorouri@mie.ie).
License
This project is licensed under the MIT License - see the LICENSE file for details.
