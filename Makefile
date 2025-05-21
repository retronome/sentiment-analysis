.PHONY: default train infer interactive

default: interactive

train:
	conda activate pytorch && \
	python src/train.py

infer:
	conda activate pytorch && \
	python src/infer.py

interactive:
	conda activate pytorch && \
	python src/infer.py --interactive
