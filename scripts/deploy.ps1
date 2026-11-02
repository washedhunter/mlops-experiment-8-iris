# Build the Docker image and deploy the API + Prometheus + Grafana to the current Kubernetes context.
# Usage (from the project root):  powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1
$ErrorActionPreference = "Stop"

docker build -t iris-mlops-api:1.0 .

kubectl apply -f k8s/api.yaml

kubectl create configmap grafana-datasources --from-file=monitoring/grafana/provisioning/datasources --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap grafana-dashboard-provider --from-file=monitoring/grafana/provisioning/dashboards --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap grafana-dashboards --from-file=monitoring/grafana/dashboards --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml

kubectl rollout status deployment/iris-api --timeout=180s
kubectl rollout status deployment/prometheus --timeout=180s
kubectl rollout status deployment/grafana --timeout=180s
kubectl get pods,svc
