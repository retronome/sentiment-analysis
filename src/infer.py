import torch
import json
import argparse
import os
import sys
import re

class SentimentClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, bidirectional=True):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        # If bidirectional, we need to multiply hidden_dim by 2
        self.fc = torch.nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.bidirectional = bidirectional

    def forward(self, text):
        # text shape: [batch size, sent len]
        embedded = self.embedding(text)
        # embedded shape: [batch size, sent len, emb dim]
        output, (hidden, cell) = self.rnn(embedded)

        # If bidirectional, concatenate the final forward and backward hidden states
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]

        return self.fc(hidden)

def load_model_and_vocab():
    """Load the trained model, vocabulary, and configuration"""
    # Check if model files exist
    if not os.path.exists('data/processed/sentiment_model.pt'):
        print("Error: Model file not found. Please train the model first with 'make train'")
        sys.exit(1)

    if not os.path.exists('data/processed/vocab.json'):
        print("Error: Vocabulary file not found. Please train the model first with 'make train'")
        sys.exit(1)

    if not os.path.exists('data/processed/model_config.json'):
        print("Error: Model configuration file not found. Please train the model first with 'make train'")
        sys.exit(1)

    # Load vocabulary
    with open('data/processed/vocab.json', 'r') as f:
        vocab = json.load(f)

    # Load model configuration
    with open('data/processed/model_config.json', 'r') as f:
        model_config = json.load(f)

    # Create model instance
    model = SentimentClassifier(
        model_config['vocab_size'],
        model_config['embedding_dim'],
        model_config['hidden_dim'],
        model_config['output_dim'],
        model_config['bidirectional']
    )

    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load model state
    model.load_state_dict(torch.load('data/processed/sentiment_model.pt', map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    return model, vocab, device

def predict_sentiment(text, model, vocab, device, max_len=100):  # Changed max_len from 50 to 100
    """Predict sentiment for a given text"""
    # Preprocess text (same as in training)
    text = preprocess_text(text)

    # Tokenize
    tokens = [vocab.get(word, 1) for word in text.lower().split()]

    # Pad or truncate
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]

    # Create tensor
    tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        prediction = model(tensor)

    # Get predicted class and confidence
    probs = torch.softmax(prediction, dim=1)
    pred_class = torch.argmax(prediction, dim=1).item()
    confidence = probs[0][pred_class].item()

    sentiment = "positive" if pred_class == 1 else "negative"

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "class": pred_class
    }

# Add the preprocess_text function from train.py
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def interactive_mode():
    """Run an interactive loop for sentiment analysis"""
    print("Loading sentiment analysis model...")
    model, vocab, device = load_model_and_vocab()

    print("\n===== Sentiment Analysis Interactive Mode =====")
    print("Enter text to analyze sentiment (type 'exit' or 'quit' to end):")

    while True:
        # Get input from user
        text = input("\n> ")

        # Check if user wants to exit
        if text.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        # Skip empty input
        if not text.strip():
            continue

        # Predict sentiment
        result = predict_sentiment(text, model, vocab, device)

        # Display result with confidence
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sentiment Analysis Inference')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()

    # If no arguments, default to interactive mode
    if not args.text and not args.interactive:
        args.interactive = True

    # Check if interactive mode
    if args.interactive:
        interactive_mode()
    else:
        # Load model and vocab
        model, vocab, device = load_model_and_vocab()

        # Predict sentiment
        result = predict_sentiment(args.text, model, vocab, device)

        # Display result
        print(f"Text: '{args.text}'")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")

if __name__ == "__main__":
    main()
