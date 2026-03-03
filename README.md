# Real-Time Structural Topology Optimization Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-FF6F00.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![GCP](https://img.shields.io/badge/GCP-Cloud_Run-4285F4.svg)](https://cloud.google.com/)

**Live Production App:** [Launch Application Here](https://topology-gui-3124106137.asia-southeast1.run.app/) 

## Project Overview
This repository hosts a production-grade Deep Learning surrogate model designed to solve Structural Topology Optimization problems in real-time. By replacing computationally expensive matrix inversions of traditional Finite Element Analysis (FEA) solvers with neural networks, the system predicts optimal load-bearing material distribution instantaneously.

The application demonstrates the architectural evolution of this engineering tool across two phases:
* **Phase 1 (Baseline):** A deterministic **U-Net Convolutional Neural Network (CNN)** rigidly trained to output structures at exactly 40% volume fraction.
* **Phase 2 (Advanced):** A **Conditional Generative Adversarial Network (Pix2Pix cGAN)** featuring a 4-channel input tensor. This introduces the *Target Volume Fraction* as a dynamic variable, allowing engineers to explore continuous mass-to-stiffness trade-offs (20% to 80% mass) in real-time.

## Key Engineering Metrics
* **Inference Speed (Phase 2):** Achieved a **~30x single-shot latency speedup** (<0.05s per design) for interactive GUIs, and a **>300x batch throughput speedup** for large-scale generative design exploration.
* **Accuracy:** Maintained **92.3% Intersection-over-Union (IoU)** against physics-grounded dataset benchmarks.
* **Data Pipeline:** Engineered a custom Python-based SIMP FEA solver to generate over 33,000 combined training samples exponentially faster than standard commercial API automation.

## Cloud & MLOps Architecture
To ensure cross-platform environment stability and high availability, this application is fully containerized and deployed via a modern serverless pipeline.

* **Machine Learning:** TensorFlow / Keras (U-Net & cGAN)
* **User Interface:** Streamlit (Featuring a multi-tab engineering dashboard with live metric tracking)
* **Environment Management:** **Docker** (Built from source to resolve underlying C++ binary conflicts between TensorFlow and NumPy)
* **Cloud Infrastructure:** **Google Cloud Platform (GCP)**
  * **Artifact Registry:** Secure, compressed container image storage co-located in `asia-southeast1` to eliminate cross-region egress latency and speed up cold starts.
  * **Cloud Run:** Serverless deployment providing automated horizontal scaling and "scale-to-zero" cost efficiency.
* **Version Control:** Managed massive model weights (`.keras` and `.h5` files) using **Git Large File Storage (LFS)**.

## How to Run Locally

Because the application is fully Dockerized, you can run the exact production environment locally without installing any Python dependencies or worrying about OS conflicts.

### Prerequisites
* Docker Desktop installed and running.
* Git LFS installed (critical for pulling the large `.keras` and `.h5` model weights).

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
   docker run -p 8080:8080 topopt-ai-app
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:8080`

---
*Developed by Junyen Chau | B.Eng Mechanical Engineering (Aeronautical Specialisation), Minor in Computing, National University of Singapore*
