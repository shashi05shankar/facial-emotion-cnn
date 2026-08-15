.PHONY: install test download-data webcam

install:
	pip install -e ".[dev]"

test:
	pytest -v

download-data:
	python scripts/download_dataset.py

webcam:
	python scripts/run_webcam_demo.py
