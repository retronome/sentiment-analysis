.PHONY: default train infer interactive

default: train interactive

train:
	python src/train.py

infer:
	python src/infer.py --text "I really enjoyed this movie, it was amazing!"

interactive:
	python src/infer.py --interactive

huggingface:
	python src/hf.py --interactive
