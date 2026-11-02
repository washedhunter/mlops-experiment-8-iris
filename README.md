# MLOps Experiment 8: Iris Species Classifier (end-to-end)

A small ML project used to show a full MLOps pipeline:

| Stage | Tool | Files |
|---|---|---|
| Model | scikit-learn RandomForest on the Iris dataset (≈97% test accuracy) | `src/train.py` |
| Serving API + metrics | FastAPI + prometheus-client (`/predict`, `/health`, `/metrics`) | `src/api.py` |
| Version control | Git + GitHub | — |
| CI/CD | GitHub Actions: tests → train → Docker build & push to GHCR | `.github/workflows/ci.yml` |
| Container | Docker | `Dockerfile` |
| Orchestration | Kubernetes (Docker Desktop): 2 API replicas + Prometheus + Grafana | `k8s/` |
| Monitoring | Prometheus scrapes each pod, Grafana dashboard is auto-provisioned | `k8s/prometheus.yaml`, `monitoring/` |
| Web app | Streamlit (deployed on Streamlit Community Cloud) | `app.py` |

## Run locally
```bash
pip install -r requirements-dev.txt
python -m src.train          # train + save model/model.joblib
pytest -v                    # unit tests
streamlit run app.py         # web UI on http://localhost:8501
```

## Docker
```bash
docker build -t iris-mlops-api:1.0 .
docker run -p 8000:8000 iris-mlops-api:1.0
```

## Kubernetes + monitoring
```powershell
powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1
kubectl port-forward svc/iris-api 8000:8000
kubectl port-forward svc/prometheus 9090:9090
kubectl port-forward svc/grafana 3000:3000
python scripts/load_test.py --url http://localhost:8000 --n 400
```
Grafana: http://localhost:3000 → Dashboards → MLOps → *Iris Model API - MLOps Monitoring* (login admin/admin).
