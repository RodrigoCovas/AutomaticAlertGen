import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import os
import random

def process_in_batches(dataset_split, dataset_name, batch_size, tokenizer, max_length, model, output_dir):
    """Processes the dataset in batches."""
    tokens = dataset_split["tokens"]
    labels = dataset_split["ner_tags"]
    sentiments = dataset_split["sentiments"]

    for i in tqdm(range(0, len(tokens), batch_size), desc=f"Processing {dataset_name} batches"):
        batch_tokens = tokens[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_sentiments = sentiments[i:i + batch_size]

        embeddings, padded_labels, processed_sentiments = process_batch(batch_tokens, batch_labels, batch_sentiments, tokenizer, max_length, model)
        save_batch(dataset_name, i // batch_size, embeddings, padded_labels, processed_sentiments, output_dir)
        
def process_batch(batch_tokens, batch_labels, batch_sentiments, tokenizer, max_length, model):
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

def save_batch(dataset_name, batch_idx, embeddings, labels, sentiments, output_dir):
    """Saves processed batch embeddings and labels to disk."""
    save_path = os.path.join(output_dir, dataset_name, f"batch_{batch_idx}.pt")
    torch.save({
        "embeddings": embeddings.cpu(),
        "labels": labels.cpu(),
        "sentiments": sentiments.cpu()
    }, save_path)

def get_max_length(dataset_split):
    """Returns the maximum length of tokens in the dataset split."""
    max_length = max(len(tokens) for tokens in dataset_split["tokens"])
    return max_length

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
    return random.shuffle([s for l in sentiments for s in l])

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

def extract_sentences(dataset_split):
    sentences = [" ".join(tokens) for tokens in dataset_split["tokens"]]
    return sentences