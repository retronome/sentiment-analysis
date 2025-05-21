# Sentiment Analysis Model

A simple sentiment analysis model built with PyTorch that demonstrates text classification on Apple Silicon hardware.

## Description

This project implements a bidirectional LSTM-based sentiment analysis model that can classify text as either positive or negative. The model leverages Metal Performance Shaders (MPS) acceleration on Apple Silicon Macs for efficient training and inference.

Features:
- Bidirectional LSTM neural network architecture with dropout
- Training on Stanford Sentiment Treebank (SST-2) dataset
- Domain-specific fine-tuning for better performance
- Vocabulary generation from training data
- Optimized for Apple Silicon via MPS acceleration

## Requirements

- Python 3.10+
- PyTorch with MPS support
- Conda/Miniconda environment

## Installation

This project uses a Conda environment. Here's how to set it up:

```bash
# Install Miniconda if you don't have it
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/conda.sh

# Create and activate the environment
conda create -n pytorch python=3.10 -y
conda activate pytorch

# Install required packages
conda install numpy=1.24 -y
conda install pytorch torchvision torchaudio -c pytorch -y
conda install scikit-learn pandas matplotlib -c conda-forge -y
conda install urllib3 -y
```

## Running the Model

To train and test the model, simply run:

```bash
# Just run make in the project directory
make
```

This will:
1. Train the sentiment analysis model on the SST-2 dataset
2. Perform domain-specific fine-tuning if needed
3. Test the model on specific examples
4. Launch the interactive mode for testing your own phrases

You can also run specific commands:

```bash
# Train only
make train

# Run interactive mode (requires model to be trained first)
make interactive

# Test a specific phrase (requires model to be trained first)
make infer
```

## How It Works

1. The script downloads the Stanford Sentiment Treebank dataset
2. It builds a vocabulary from the dataset
3. The data is split into training and validation sets
4. A bidirectional LSTM model with dropout is trained for sentiment classification
5. Domain-specific examples are used for fine-tuning if necessary
6. The best model is saved to `data/processed/sentiment_model.pt`
7. The model is tested on specific examples to verify performance
8. Interactive mode allows testing your own phrases

## Model Architecture

The model uses:
- An embedding layer to convert words to vectors
- A bidirectional LSTM layer with dropout to capture sequential information
- A linear layer for final classification

## Using the Model for Inference

After training, you can use the model for sentiment analysis on new text:

```python
# Load the model
model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('data/processed/sentiment_model.pt', weights_only=True))
model.eval()

# Load vocabulary
with open('data/processed/vocab.json', 'r') as f:
    vocab = json.load(f)

# Preprocess text
text = text.lower()
text = ''.join([c for c in text if c.isalpha() or c.isspace()])
text = ' '.join(text.split())

# Tokenize
tokens = [vocab.get(word, 1) for word in text.split()]
if len(tokens) < 50:
    tokens = tokens + [0] * (50 - len(tokens))
else:
    tokens = tokens[:50]

# Get prediction
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
tensor = torch.tensor(tokens).unsqueeze(0).to(device)
prediction = model(tensor)
probs = torch.softmax(prediction, dim=1)
pred_class = torch.argmax(prediction, dim=1).item()
confidence = probs[0][pred_class].item()
sentiment = "positive" if pred_class == 1 else "negative"
print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")
```

## Limitations

- The model performs binary sentiment classification only (positive/negative)
- It may struggle with sarcasm, irony, and complex emotional expressions
- The vocabulary is limited to words seen during training

## Extending the Project

To improve the model:
- Add more training data from diverse sources
- Implement more advanced preprocessing (stemming, lemmatization)
- Try different model architectures (BERT, transformers)
- Add support for more sentiment classes (neutral, mixed, etc.)
- Train with emoji and emoticon understanding

## License

MIT
