# SmartPark-Vision OEL 🚗
Real-Time Automated Parking Lot Occupancy Detection & Vehicle Counting via Thresholding, Contours, and Lightweight YOLOv8.

**Submitted By:** Faryan Afaq (Roll No: 64483) | BS(AI)  
**Institution:** Iqra University 

## Project Overview
This repository contains the implementation of a dual-pipeline computer vision system designed to monitor vehicle occupancy and detect incoming traffic from CCTV feeds. 

## Performance Benchmark (CPU)
| Detection Approach | Inference Speed | Accuracy |
| :--- | :--- | :--- |
| **Classical Thresholding + Pixel Counting** | ~88.98 FPS | ~82% |
| **Deep Learning (YOLOv8n)** | ~9.85 FPS | ~98% |

## Repository Structure
* `/classical_detector`: Scripts for dynamic ROI extraction, adaptive thresholding, and morphological operations.
* `/yolo_detector`: YOLOv8n deployment script for deep object detection.
* `/notebooks`: Contains the complete master pipeline (`.ipynb`).
