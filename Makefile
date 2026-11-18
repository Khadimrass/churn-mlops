.PHONY: help install data train serve test lint drift simulate simulate-drift demo mlflow-ui docker-up docker-train docker-down clean

PY ?= python
export MLFLOW_DISABLE_AGENT_HINT=1

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Installe les dépendances de dev + le package
	$(PY) -m pip install -r requirements/dev.txt
	$(PY) -m pip install -e .

data:  ## Télécharge le dataset
	$(PY) scripts/download_data.py

train: data  ## Entraîne, évalue et promeut le modèle (quality gate)
	$(PY) -m churn.train

serve:  ## Lance l'API en local (http://localhost:8000/docs)
	uvicorn churn.api.main:app --reload --port 8000

test:  ## Tests + couverture
	$(PY) -m pytest --cov=churn --cov-report=term-missing

lint:  ## Analyse statique
	ruff check src tests scripts

simulate:  ## Envoie 500 clients « normaux » à l'API
	$(PY) scripts/simulate_traffic.py --n 500

simulate-drift:  ## Envoie 500 clients avec dérive simulée
	$(PY) scripts/simulate_traffic.py --n 500 --drift

drift:  ## Rapport de dérive sur les 500 dernières prédictions (reports/drift_report.html)
	-$(PY) -m churn.monitoring.drift --last-n 500

mlflow-ui:  ## Interface MLflow locale (http://localhost:5000)
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-up: train  ## Démarre la stack complète (API, MLflow, Prometheus, Grafana)
	docker compose up -d --build

docker-train:  ## Réentraîne dans Docker via le serveur MLflow, puis recharge l'API
	docker compose --profile train run --rm trainer
	curl -fsS -X POST http://localhost:8000/model/reload

docker-down:  ## Arrête la stack
	docker compose down

clean:  ## Supprime les artefacts générés
	rm -rf mlruns mlartifacts mlflow.db models/* reports data/production .pytest_cache .ruff_cache .coverage
