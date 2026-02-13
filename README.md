# Real-Time Structural Topology Optimization Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-FF6F00.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![GCP](https://img.shields.io/badge/GCP-Cloud_Run-4285F4.svg)](https://cloud.google.com/)

**Live Production App:** [Launch Application Here](https://topopt-app-3124106137.asia-southeast1.run.app/)

## Project Overview
This repository hosts a production-grade Deep Learning surrogate model designed to solve Structural Topology Optimization problems in real-time. By utilizing a **U-Net Convolutional Neural Network (CNN)**, the system predicts optimal load-bearing material distribution, bypassing the computationally expensive iterations of traditional Finite Element Analysis (FEA) solvers.

This tool is designed to provide aeronautical and mechanical engineers with instantaneous structural feedback during the preliminary design phase.

## Key Engineering Metrics
* **Accuracy:** Achieved **92.3% Intersection-over-Union (IoU)** against physics-grounded dataset benchmarks.
* **Latency:** Reduced inference time to **~0.1s per design**, representing a **6.6x speedup** over commercial iterative solvers.
* **Data Pipeline:** Engineered a custom Python-based FEA solver to generate training samples 20x faster than standard API automation.

## Cloud & MLOps Architecture
To ensure cross-platform environment stability and high availability, this application is fully containerized and deployed via a modern serverless pipeline.

* **Machine Learning:** TensorFlow / Keras (U-Net)
* **User Interface:** Streamlit 
* **Environment Management:** **Docker** (Built from source to resolve underlying C++ binary conflicts between TensorFlow and NumPy)
* **Cloud Infrastructure:** **Google Cloud Platform (GCP)**
  * **Artifact Registry:** Secure, compressed container image storage (`asia-southeast1`).
  * **Cloud Run:** Serverless deployment providing automated horizontal scaling and "scale-to-zero" cost efficiency.
* **Version Control:** Managed massive model weights (154+ MB) using **Git Large File Storage (LFS)**.

## How to Run Locally

Because the application is fully Dockerized, you can run the exact production environment locally without installing any Python dependencies or worrying about OS conflicts.

### Prerequisites
* Docker Desktop installed and running.
* Git LFS installed (to pull the `.keras` model weights).

### Quick Start
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/junyenchau/topopt-ai-cloud.git](https://github.com/junyenchau/topopt-ai-cloud.git)
   cd topopt-ai-cloud
   ```

2. **Build the Docker container:**
   ```bash
   docker build -t topopt-ai-app .
   ```

3. **Run the container:**
   ```bash
   docker run -p 8501:8501 topopt-ai-app
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

---
*Developed by Junyen Chau | B.Eng Mechanical Engineering (Aeronautical Specialisation), Minor in Computing, National University of Singapore*
