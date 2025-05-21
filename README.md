# Sentiment Analysis Model

A simple sentiment analysis model built with PyTorch that demonstrates text classification on Apple Silicon hardware.

## Description

This project implements a lightweight LSTM-based sentiment analysis model that can classify text as either positive or negative. The model leverages Metal Performance Shaders (MPS) acceleration on Apple Silicon Macs for efficient training and inference.

Features:
- Bidirectional LSTM neural network architecture
- Vocabulary generation from training data
- Model export to ONNX format
- Optimized for Apple Silicon via MPS

## Requirements

- Python 3.10+
- PyTorch with MPS support
- Conda/Miniconda environment

## Setup

This project uses a Conda environment. If you haven't already set up the environment:

```bash
# Create and activate the PyTorch environment
conda create -n pytorch python=3.10
conda activate pytorch

# Install required packages
conda install pytorch torchvision torchaudio -c pytorch
conda install pandas matplotlib scikit-learn -c conda-forge
```

## Running the Model

To train and test the model:

```bash
# Activate the environment if not already active
conda activate pytorch

# Run the model
make
```

Alternatively, you can run it directly:

```bash
python src/main.py
```

## How It Works

1. The script creates a small dataset of positive and negative text samples
2. It builds a simple vocabulary from these samples
3. The data is split into training and validation sets
4. A bidirectional LSTM model is trained to classify sentiments
5. The best model is saved to `data/processed/sentiment_model.pt`
6. The model is exported to ONNX format for compatibility with other tools
7. The script performs inference on test sentences to demonstrate usage

## Model Architecture

The model uses:
- An embedding layer to convert words to vectors
- A bidirectional LSTM layer to capture sequential information
- A linear layer for classification

## Using the Model for Inference

After training, you can use the model for sentiment analysis on new text:

```python
# Load the model
model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('data/processed/sentiment_model.pt'))
model.eval()

# Load vocabulary
with open('data/processed/vocab.json', 'r') as f:
    vocab = json.load(f)

# Process text
text = "your text here"
tokens = [vocab.get(word, 1) for word in text.lower().split()]
if len(tokens) < 50:
    tokens = tokens + [0] * (50 - len(tokens))
else:
    tokens = tokens[:50]

# Get prediction
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
tensor = torch.tensor(tokens).unsqueeze(0).to(device)
prediction = model(tensor)
pred_class = torch.argmax(prediction, dim=1).item()
sentiment = "positive" if pred_class == 1 else "negative"
print(f"Sentiment: {sentiment}")
```

## Extending the Project

To improve the model:
- Add more training data
- Implement more advanced preprocessing (tokenization, stemming)
- Try different model architectures (Transformer-based models)
- Add support for more sentiment classes beyond binary positive/negative

## License

MIT
