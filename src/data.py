import os
from datasets import load_dataset, load_from_disk
from transformers import pipeline
from tqdm import tqdm
from transformers import BertTokenizer

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
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Define a custom sentiment analysis function with a threshold for neutrality
    def custom_sentiment_analysis(sentiment_analyzer, sentence, neutral_threshold=0.95):
        result = sentiment_analyzer(sentence)[0]  # Get the prediction
        label = result["label"]
        score = result["score"]

        # Apply threshold logic
        if score < neutral_threshold:
            label = 1
        elif label == "POSITIVE":
            label = 0
        elif label == "NEGATIVE":
            label = 2

        return {"label": label, "score": score}

    # Analyze sentiments with a progress bar and apply custom logic
    def analyze_with_progress(sentences, sentiment_analyzer):

        sentiments:list[dict] = [[], [], []]
        for sentence in tqdm(sentences, desc="Analyzing Sentiments"):
            s = custom_sentiment_analysis(sentiment_analyzer, sentence)
            sentiments[s["label"]].append(s)
            
        # Since the dataset is imbalanced (there are more positive
        # entries), We will eliminate part of the positive reviews,
        # so that they are equal in number to the negative ones.
        # (Prioritizing removing those with low score).
        if len(sentiments[0]) > len(sentiments[2]):
            sentiments[0] = sorted(
                    sentiments[0],
                    key=lambda x: x['score'], 
                    reverse=True
                )[:len(sentiments[2])]

        return [s for l in sentiments for s in l]

    train_sentiments = analyze_with_progress(train_sentences, sentiment_analyzer)
    validation_sentiments = analyze_with_progress(
        validation_sentences, sentiment_analyzer
    )
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

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    datasets_names = ["train", "validation", "test"]
    processed_dataset = {}

    for dataset_name in datasets_names:
        group_words = dataset[dataset_name]["tokens"]
        group_labels = dataset[dataset_name]["ner_tags"]

        # Initialize lists for subtokens and propagated labels
        list_tokens = []
        list_labels = []

        # Process each sentence (words + labels)
        for words, word_labels in tqdm(zip(group_words, group_labels), total=len(group_words), desc=f"Processing {dataset_name}"):
            if len(words) != len(word_labels):
                raise ValueError(f"Mismatch between words and labels in {dataset_name}: {words}, {word_labels}")
            
            tokens = []
            labels = []
            for word, label in zip(words, word_labels):
                # Tokenize the word into subtokens
                word_tokens = tokenizer.tokenize(word)
                # Extend the token list with subtokens
                tokens.extend(word_tokens)
                # Extend the label list with the same label for each subtoken
                labels.extend([label] * len(word_tokens))
            
            list_tokens.append(tokens)
            list_labels.append(labels)
        
        processed_dataset[dataset_name] = {
            "tokens": list_tokens,
            "ner_tags": list_labels
        }
    
    # Save the modified dataset to a directory on disk
    print("Saving processed dataset to disk...")
    dataset.save_to_disk(dataset_path)