import os
from datasets import load_dataset, load_from_disk
from transformers import pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define the path to save/load the dataset
dataset_path = "data/conll2003_dataset"

# Check if the dataset has already been processed and saved
if os.path.exists(dataset_path):
    print("Loading dataset from disk...")
    dataset = load_from_disk(dataset_path)
else:
    print("Processing dataset...")

    # Load the original CoNLL-2003 dataset
    dataset = load_dataset("conll2003", trust_remote_code=True)

    # Extract sentences from tokens
    def extract_sentences(dataset_split):
        sentences = [" ".join(tokens) for tokens in dataset_split["tokens"]]
        return sentences

    train_sentences = extract_sentences(dataset["train"])
    validation_sentences = extract_sentences(dataset["validation"])
    test_sentences = extract_sentences(dataset["test"])

    # Load the sentiment-analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Define a custom sentiment analysis function with a threshold for neutrality
    def custom_sentiment_analysis(sentiment_analyzer, sentence, neutral_threshold=0.7):
        result = sentiment_analyzer(sentence)[0]  # Get the prediction
        label = result["label"]
        score = result["score"]
        
        # Apply threshold logic
        if score < neutral_threshold:
            label = "NEUTRAL"
        
        return {"label": label, "score": score}

    # Analyze sentiments with a progress bar and apply custom logic
    def analyze_with_progress(sentences, sentiment_analyzer, neutral_threshold=0.7):
        sentiments = []
        for sentence in tqdm(sentences, desc="Analyzing Sentiments"):
            sentiments.append(custom_sentiment_analysis(sentiment_analyzer, sentence, neutral_threshold))
        return sentiments

    train_sentiments = analyze_with_progress(train_sentences, sentiment_analyzer)
    validation_sentiments = analyze_with_progress(validation_sentences, sentiment_analyzer)
    test_sentiments = analyze_with_progress(test_sentences, sentiment_analyzer)

    # Add sentiments and scores to the dataset
    def add_sentiments(dataset_split, sentiments):
        # Add 'sentiments' column with labels (e.g., "POSITIVE", "NEGATIVE", or "NEUTRAL")
        labels = [result["label"] for result in sentiments]
        # Add 'scores' column with confidence scores
        scores = [result["score"] for result in sentiments]
        
        # Add both columns to the dataset split
        dataset_split = dataset_split.add_column("sentiments", labels)
        dataset_split = dataset_split.add_column("scores", scores)
        
        return dataset_split

    dataset["train"] = add_sentiments(dataset["train"], train_sentiments)
    dataset["validation"] = add_sentiments(dataset["validation"], validation_sentiments)
    dataset["test"] = add_sentiments(dataset["test"], test_sentiments)

    # Save the modified dataset to a directory on disk
    print("Saving processed dataset to disk...")
    dataset.save_to_disk(dataset_path)

# Convert datasets to PyTorch format (optional)
train_dataset = dataset["train"].with_format("torch")
val_dataset = dataset["validation"].with_format("torch")
test_dataset = dataset["test"].with_format("torch")

print(train_dataset)
print(val_dataset)
print(test_dataset)
