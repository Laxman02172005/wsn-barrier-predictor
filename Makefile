.PHONY: install train api-run streamlit-run docker-build docker-run mlflow test

install:
	pip install -r requirements.txt

train:
	python -m src.train

api-run:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

streamlit-run:
	streamlit run streamlit_app/app.py

docker-build:
	docker build -t wsn-api:latest -f Dockerfile.api .
	docker build -t wsn-streamlit:latest -f Dockerfile.streamlit .

docker-run:
	docker-compose up --build

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

test:
	pytest tests/
