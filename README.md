
## Multi-Model Fraud Detection System

A robust, real-time fraud detection platform that combines multiple machine learning architectures — XGBoost, Graph Neural Networks (GNN), and Transformers — to identify and prevent fraudulent transactions at scale. The system simulates real-world data streams and employs modern MLOps practices for deployment, monitoring, and scalability.

⸻

## Overview

This project focuses on detecting fraudulent activities in financial transactions by leveraging a multi-model ensemble approach. It integrates both structured (tabular) and unstructured (graph/text) data, ensuring high accuracy and adaptability in dynamic environments.

⸻

## Key Features
	•	Real-time Fraud Detection using Apache Kafka & Flink for streaming data ingestion and processing.
	•	Hybrid ML Architecture combining XGBoost, GNN, and Transformer models for superior prediction performance.
	•	Feature Store Integration with Feast to maintain consistent features across training and inference.
	•	Model Drift Monitoring using EvidentlyAI for continuous model health tracking.
	•	Scalable Deployment via Docker and Kubernetes with canary release strategies.
	•	CI/CD Automation using MLflow and GitHub Actions for versioning and seamless updates.

⸻

## Architecture

Data Stream (Kafka) 
        ↓
Preprocessing & Feature Engineering (Flink, Spark)
        ↓
Feature Store (Feast)
        ↓
Model Ensemble:
   ├── XGBoost (Tabular Features)
   ├── GNN (Transaction Graphs)
   └── Transformer (Text/Sequential Patterns)
        ↓
Model Scoring & Drift Detection (EvidentlyAI)
        ↓
Dashboard & Alerts (Streamlit / Grafana)


## Tech Stack

Languages: Python, SQL
Frameworks: PyTorch, XGBoost, Transformers, DGL (for GNN)
Data Pipeline: Apache Kafka, Flink, Spark
MLOps: MLflow, Docker, Kubernetes, EvidentlyAI, Feast
Cloud: AWS (S3, Lambda, EKS) / Azure / GCP (Vertex AI)
Visualization: Streamlit, Power BI

⸻

## Model Workflow
	1.	Data Ingestion: Stream transaction data through Kafka topics.
	2.	Data Transformation: Use Flink to process, clean, and aggregate features in real-time.
	3.	Feature Storage: Store consistent features with Feast for training/inference parity.
	4.	Model Training:
	•	XGBoost handles numerical transaction data.
	•	GNN models relational dependencies (e.g., customer-to-customer links).
	•	Transformers capture sequential behavior and temporal trends.
	5.	Model Ensemble: Combine model outputs through a meta-learner for final prediction.
	6.	Monitoring: Track model drift, latency, and accuracy with EvidentlyAI and Grafana dashboards.

⸻

## Setup & Installation

# Clone the repo
git clone https://github.com/<your-username>/multi-model-fraud-detection.git
cd multi-model-fraud-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # (use venv\Scripts\activate on Windows)

# Install dependencies
pip install -r requirements.txt

# Start Kafka and Flink services (Docker-based setup)
docker-compose up -d

# Run model pipeline
python main.py


⸻

## Results
	•	Improved detection accuracy compared to standalone models.
	•	Reduced false positives through ensemble voting strategy.
	•	Real-time prediction latency.
⸻

## Monitoring Dashboard

A Streamlit dashboard displays live metrics for:
	•	Data flow and latency
	•	Fraud probability scores
	•	Model drift and feature stability
	•	Ensemble performance comparison

⸻

## Author

Pavan Kalyan Reddy Burgapally
AI/ML Engineer | Data Scientist
📧 pawankalyanburgapally@gmail.com
Linkedin: https://www.linkedin.com/in/pavan-kalyan-b-a91505276

