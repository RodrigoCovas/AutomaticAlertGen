from tqdm import tqdm
import torch
import os
import random
import numpy as np
from torch.jit import RecursiveScriptModule
from typing import Optional

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

    # Convert labels to tensors and pad them to max_length
    label_tensors = [torch.tensor(labels) for labels in batch_labels]
    padded_labels = torch.full((len(label_tensors), max_length), -1)  # Initialize with padding value
    for idx, label_tensor in enumerate(label_tensors):
        padded_labels[idx, :len(label_tensor)] = label_tensor  # Copy label values

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
    # for sentence in sentences:
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
    elif len(sentiments[0]) < len(sentiments[2]):
        for s in sorted(
                    sentiments[2],
                    key=lambda x: x['score'], 
                    reverse=True
                )[:len(sentiments[0])]:
            s['drop_row'] = True
    shuffled_list = [s for l in sentiments for s in l]
    random.shuffle(shuffled_list)
    return shuffled_list

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

class MAE:
    """
    This class is the MAE (Mean Absolute Error) object.

    Attr:
        total_error: cumulative absolute error across all predictions.
        total: number of total examples.
    """

    total_error: float
    total: int

    def __init__(self) -> None:
        """
        This is the constructor of the MeanAbsoluteError class. It initializes
        total_error and total to zero.
        """
        self.total_error: float = 0.0
        self.total: int = 0

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """
        This method updates the cumulative absolute error and the total count.

        Args:
            predictions: outputs of the model. Dimensions: [batch, ...].
            labels: ground truth values. Dimensions: [batch, ...].
        """
        assert (
            predictions.shape == labels.shape
        ), f"Shape mismatch between predictions and labels: {predictions.shape} vs {labels.shape}"
        batch_error: float = torch.abs(predictions - labels).sum().item()
        self.total_error += batch_error
        self.total += labels.numel()

    def compute(self) -> float:
        """
        This method computes and returns the Mean Absolute Error (MAE).

        Returns:
            MAE value.
        """
        if self.total == 0:
            return 0.0
        return self.total_error / self.total

    def reset(self) -> None:
        """
        This method resets the cumulative error and total count to zero.
        """
        self.total_error = 0.0
        self.total = 0


class EarlyStopping:
    """
    Custom implementation of early stopping to halt training when validation performance
    stops improving.
    """

    patience: int
    delta: float
    verbose: bool
    mode: str
    best_score: Optional[float]
    counter: int
    early_stop: bool

    def __init__(
        self,
        patience: int = 10,
        delta: float = 0.0,
        verbose: bool = False,
        mode: str = "min",
    ) -> None:
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            delta (float): Minimum change in monitored value to qualify as improvement.
            verbose (bool): If True, prints a message when training stops.
            mode (str): Whether to monitor for a decrease ("min") or an increase ("max").
                        Defaults to "min".
        """
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience: int = patience
        self.delta: float = delta
        self.verbose: bool = verbose
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.mode: str = mode

    def __call__(self, val_metric: float) -> None:
        """
        Checks if training should stop based on validation metric.

        Args:
            val_metric (float): Current value of the monitored validation metric.
        """
        if self.best_score is None:
            self.best_score = val_metric
        else:
            if self.mode == "min":
                if val_metric < self.best_score - self.delta:
                    self.best_score = val_metric
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.verbose:
                        print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            elif self.mode == "max":
                if val_metric > self.best_score + self.delta:
                    self.best_score = val_metric
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.verbose:
                        print(f"EarlyStopping counter: {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            if self.verbose:
                print("Early stopping triggered.")
            self.early_stop = True

def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix randomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None