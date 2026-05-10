# Wireless Sensor Network (WSN) - MLOps Project

This project predicts the "Number of Barriers" in a Wireless Sensor Network based on Area, Sensing Range, Transmission Range, and Number of Sensor Nodes.

It demonstrates a full production-grade MLOps architecture including:
- **FastAPI** for model inference
- **Streamlit** for dashboarding and Model Comparison
- **MLflow** for experiment tracking
- **SHAP** for model explainability
- **Docker** & **Kubernetes** for deployment
- **GitHub Actions** for CI/CD

## Project Structure
```text
.
├── data/               # Place data.csv here
├── models/             # Trained artifacts (model, scaler, metadata)
├── logs/               # Application logs
├── notebooks/          # Original Jupyter notebooks
├── src/                # Backend API and ML logic
├── streamlit_app/      # Frontend application
├── k8s/                # Kubernetes manifests
├── .github/workflows/  # CI/CD pipelines
├── Makefile            # Utility commands
└── docker-compose.yml  # Local Docker setup
```

## Quickstart

### 1. Prerequisites
- Python 3.10+
- Docker & Docker Compose

### 2. Setup
1. Clone the repository.
2. Place the original `data.csv` into the `data/` folder.
3. Install requirements:
   ```bash
   make install
   ```

### 3. Training
Train the models (Linear Regression, Random Forest, Gradient Boosting) and generate artifacts:
```bash
make train
```

### 4. Running Locally
Run the MLflow tracking server, FastAPI backend, and Streamlit frontend.

**Using Makefile (separate terminals):**
```bash
make mlflow
make api-run
make streamlit-run
```

**Using Docker Compose:**
```bash
make docker-run
```

### 5. Kubernetes Deployment
Apply the manifests to your cluster:
```bash
kubectl apply -f k8s/
```
