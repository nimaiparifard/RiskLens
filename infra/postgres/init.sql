-- extra databases on the shared Postgres instance
CREATE DATABASE mlflow;
CREATE DATABASE optuna;
\c risklens
CREATE EXTENSION IF NOT EXISTS vector;
