.PHONY: setup run

setup:
	python -m venv .venv
	.venv/Scripts/pip install -r requirements.txt

run:
	.venv/Scripts/python data_downloader.py
	.venv/Scripts/python evaluate.py