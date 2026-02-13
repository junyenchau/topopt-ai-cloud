# Structural AI Inference Engine (Topology Optimization)

[![Live Demo](https://img.shields.io/badge/GCP-Live_App-blue?logo=google-cloud)](YOUR_CLOUD_RUN_URL_HERE)

## Project Overview
This repository hosts a Deep Learning-based surrogate model for **Structural Topology Optimization**. [cite_start]By leveraging a **U-Net Convolutional Neural Network (CNN)**, the platform predicts optimal load-bearing structures in real-time, bypassing the high computational cost of traditional iterative FEA solvers.

### Key Engineering Results
* [cite_start]**Accuracy:** Achieved **92.3% Intersection-over-Union (IoU)** against physics-grounded benchmarks[cite: 32].
* [cite_start]**Latency:** Reduced inference time to **0.1s**, representing a **6.6x speedup** over commercial iterative solvers[cite: 34].
* [cite_start]**Data Engineering:** Developed a custom solver generating training samples **20x faster** than standard API automation[cite: 33].

## Tech Stack & MLOps
To ensure enterprise-grade reliability and scalability, the application is built using a modern MLOps pipeline:

* **Framework:** TensorFlow / Keras (U-Net Architecture)
* **Frontend:** Streamlit
* **Containerization:** **Docker** (to resolve complex C++ binary conflicts between TensorFlow and NumPy)
* **Cloud Infrastructure:** **Google Cloud Platform (GCP)**
  * **Artifact Registry:** Secure storage for compressed container images.
  * **Cloud Run:** Serverless deployment providing automated scaling and "scale-to-zero" cost efficiency.

## How to Run Locally
Ensure you have Docker installed, then run:
```bash
docker build -t topopt-ai-app .
docker run -p 8501:8501 topopt-ai-app
