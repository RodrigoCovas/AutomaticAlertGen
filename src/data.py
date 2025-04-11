import os
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset, DataLoader

# Define the path to save/load the dataset
dataset_path = "data"

# Check if the dataset has already been processed and saved
if os.path.exists(dataset_path):
    class CustomPTDataset(Dataset):
        def __init__(self, directory):
            """
            Args:
                directory (str): Path to the directory containing .pt files.
            """
            self.directory = directory
            self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]
            self.file_paths.sort()  # Ensure consistent ordering

        def __len__(self):
            """Returns the total number of .pt files."""
            return len(self.file_paths)

        def __getitem__(self, idx):
            """
            Args:
                idx (int): Index of the file to load.
            Returns:
                Tensor: Loaded tensor from the .pt file.
            """
            file_path = self.file_paths[idx]
            data = torch.load(file_path, weights_only=True)
            return data
    
    # Paths to your dataset directories
    train_dir = f"{dataset_path}/train"
    val_dir = f"{dataset_path}/validation"
    test_dir = f"{dataset_path}/test"

    # Create datasets
    train_dataset = CustomPTDataset(train_dir)
    val_dataset = CustomPTDataset(val_dir)
    test_dataset = CustomPTDataset(test_dir)
    
    batch_size = 32  # Set your desired batch size

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Use next iter to print the first batch of data
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    test_iter = iter(test_loader)
    train_batch = next(train_iter)
    val_batch = next(val_iter)
    test_batch = next(test_iter)
    print("Train batch:", train_batch)
    print("Validation batch:", val_batch)
    print("Test batch:", test_batch)


else:
    print("Processing dataset...")

    # Load the original CoNLL-2003 dataset
    dataset = load_dataset("conll2003", trust_remote_code=True)
    dataset["train"] = dataset["train"].select(range(300))
    dataset["validation"] = dataset["validation"].select(range(300))
    dataset["test"] = dataset["test"].select(range(300))

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
    def analyze_with_progress(sentences, sentiment_analyzer) -> list[dict]:

        sentiments:list[dict] = [[], [], []]
        for sentence in tqdm(sentences, desc="Analyzing Sentiments"):
            s = custom_sentiment_analysis(sentiment_analyzer, sentence)
            s['sentence'] = sentence # Saving the original sentence in s so as to retrieve it later
            s['drop_row'] = False
            sentiments[s["label"]].append(s)
            
        # Since the dataset is imbalanced (there are more positive
        # entries), We will eliminate part of the positive reviews,
        # so that they are equal in number to the negative ones.
        # (Prioritizing removing those with low score).
        if len(sentiments[0]) > len(sentiments[2]):
            for s in sorted(
                        sentiments[0],
                        key=lambda x: x['score'], 
                        reverse=True
                    )[:len(sentiments[2])]:
                s['drop_row'] = True
        return [s for l in sentiments for s in l]
    
    train_sentiments = analyze_with_progress(train_sentences, sentiment_analyzer)
    validation_sentiments = analyze_with_progress(validation_sentences, sentiment_analyzer)
    test_sentiments = analyze_with_progress(test_sentences, sentiment_analyzer)

    # Add sentiments and scores to the dataset
    def add_sentiments(dataset_split, sentiments):
        # Add 'sentiments' column with labels (e.g., "POSITIVE", "NEGATIVE", or "NEUTRAL")
        labels = [result["label"] for result in sentiments]
        # Add 'scores' column with confidence scores
        scores = [result["score"] for result in sentiments]
        drop_row = [result["drop_row"] for result in sentiments]
        
        # Add both columns to the dataset split
        dataset_split = dataset_split.add_column("sentiments", labels)
        dataset_split = dataset_split.add_column("scores", scores)
        dataset_split = dataset_split.add_column("drop_row", drop_row)
        
        return dataset_split.filter(lambda x: not x['drop_row']).remove_columns(["drop_row"])
    
    dataset["train"] = add_sentiments(dataset["train"], train_sentiments)
    dataset["validation"] = add_sentiments(dataset["validation"], validation_sentiments)
    dataset["test"] = add_sentiments(dataset["test"], test_sentiments)

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained('bert-base-uncased')
    
    def get_max_length(dataset_split):
        """Returns the maximum length of tokens in the dataset split."""
        max_length = max(len(tokens) for tokens in dataset_split["tokens"])
        return max_length
    
    length_list = []
    for dataset_name in ["train", "validation", "test"]:
        length_list.append(get_max_length(dataset[dataset_name]))
    max_length = max(length_list) + 2  # Adding 2 for [CLS] and [SEP] tokens
    print(f"Max length of tokens: {max_length}")
    
    datasets_names = ["train", "validation", "test"]
    processed_dataset = {}
    
    # Define batch size
    BATCH_SIZE = 100

    # Directory to save processed batches
    output_dir = "./data"
    datasets_names = ["train", "validation", "test"]
    for dataset_name in datasets_names:
        os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)

    def process_batch(batch_tokens, batch_labels, batch_sentiments):
        """Processes a single batch of tokens and labels."""
        # Tokenize and encode the batch
        encoded_batch = tokenizer(
            batch_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True  # Add [CLS] and [SEP] tokens
        )
        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch["attention_mask"]

        # Generate embeddings using BERT
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Convert labels to tensors and pad them dynamically
        label_tensors = [torch.tensor(labels) for labels in batch_labels]
        padded_labels = pad_sequence(label_tensors, batch_first=True, padding_value=-1)
        sentiments = torch.tensor(batch_sentiments)

        return embeddings, padded_labels, sentiments

    def save_batch(dataset_name, batch_idx, embeddings, labels, sentiments):
        """Saves processed batch embeddings and labels to disk."""
        save_path = os.path.join(output_dir, dataset_name, f"batch_{batch_idx}.pt")
        torch.save({
            "embeddings": embeddings.cpu(),
            "labels": labels.cpu(),
            "sentiments": sentiments.cpu()
        }, save_path)

    # Process dataset in batches for each split (train/validation/test)
    def process_in_batches(dataset_split, dataset_name):
        """Processes the dataset in batches."""
        tokens = dataset_split["tokens"]
        labels = dataset_split["ner_tags"]
        sentiments = dataset_split["sentiments"]

        for i in tqdm(range(0, len(tokens), BATCH_SIZE), desc=f"Processing {dataset_name} batches"):
            batch_tokens = tokens[i:i + BATCH_SIZE]
            batch_labels = labels[i:i + BATCH_SIZE]
            batch_sentiments = sentiments[i:i + BATCH_SIZE]

            embeddings, padded_labels, processed_sentiments = process_batch(batch_tokens, batch_labels, batch_sentiments)
            save_batch(dataset_name, i // BATCH_SIZE, embeddings, padded_labels, processed_sentiments)

    # Process all splits (train/validation/test)
    for dataset_name in datasets_names:
        print(f"Processing {dataset_name} split...")
        process_in_batches(dataset[dataset_name], dataset_name)

    print(f"Processed batches saved to {output_dir}")