import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os

def load_model_and_tokenizer():
    """Load pre-trained sentiment analysis model and tokenizer from Hugging Face"""
    print("Loading pre-trained sentiment analysis model...")

    # Use a small, efficient model for sentiment analysis
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Use MPS (Metal Performance Shaders) on Apple Silicon if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    return model, tokenizer, device

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for a given text"""
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class and confidence
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    pred_class = torch.argmax(logits, dim=1).item()
    confidence = probabilities[0][pred_class].item()

    # Map class to sentiment (depends on the model, for distilbert-sst2: 0=negative, 1=positive)
    sentiment = "positive" if pred_class == 1 else "negative"

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "class": pred_class
    }

def interactive_mode():
    """Run an interactive loop for sentiment analysis"""
    model, tokenizer, device = load_model_and_tokenizer()

    print("\n===== Sentiment Analysis Interactive Mode (Hugging Face) =====")
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
        result = predict_sentiment(text, model, tokenizer, device)

        # Display result with confidence
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sentiment Analysis using Hugging Face')
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
        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer()

        # Predict sentiment
        result = predict_sentiment(args.text, model, tokenizer, device)

        # Display result
        print(f"Text: '{args.text}'")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")

if __name__ == "__main__":
    main()
