.PHONY: install train test clean docker

install:
	pip install -r requirements.txt

train:
	python main.py

test:
	python -m pytest tests/ -v

clean:
	rm -rf checkpoints/ lora-adapter/ __pycache__/ src/__pycache__/ tests/__pycache__/

docker:
	docker build -t lora-fine-tuning .
