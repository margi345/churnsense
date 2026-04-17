.PHONY: install clean train test api dashboard

install:
	pip install -r requirements.txt

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

train:
	python -m src.data.cleaner
	python -m src.data.features
	python -m src.models.train
	python -m src.models.survival
	python -m src.explainability.segments

test:
	pytest tests/ -v

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py