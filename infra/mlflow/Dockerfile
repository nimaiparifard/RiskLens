FROM python:3.12-slim
RUN pip install --no-cache-dir "mlflow>=2.14" psycopg2-binary boto3 && useradd --create-home --uid 10001 mlflow
USER mlflow
EXPOSE 5000
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} --artifacts-destination ${MLFLOW_ARTIFACTS_DESTINATION} --serve-artifacts --allowed-hosts '*'"]
