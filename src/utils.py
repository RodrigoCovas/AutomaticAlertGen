from tqdm import tqdm
import torch
import os
import random
import numpy as np
from torch.jit import RecursiveScriptModule
from typing import Optional, List, Dict, Tuple, Union
from transformers import PreTrainedTokenizerBase


def compute_class_weights(
    train_loader: torch.utils.data.DataLoader, num_classes: int
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets, useful for loss weighting.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: Normalized class weights tensor.
    """
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    for batch in train_loader:
        labels = batch["labels"].view(-1)
        mask = labels != -1  # Exclude padding (for sequence tasks)
        labels = labels[mask]
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum()
    # Avoid division by zero
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    class_weights = class_weights * (num_classes / class_weights.sum())  # Normalize
    return class_weights


def process_in_batches(
    dataset_split: Dict[str, List[Union[List[str], int]]],
    dataset_name: str,
    batch_size: int,
    tokenizer: object,
    max_length: int,
    model: torch.nn.Module,
    output_dir: str,
    task: str = "ner",
) -> None:
    """
    Processes a dataset split in batches, computes embeddings, pads labels, and
    saves batches to disk.

    Args:
        dataset_split (dict): Dictionary with split data (tokens/sentences and labels).
        dataset_name (str): Name of the dataset split (e.g., 'train').
        batch_size (int): Batch size for processing.
        tokenizer (object): Tokenizer to encode text.
        max_length (int): Maximum sequence length for padding/truncation.
        model (torch.nn.Module): Model to produce embeddings.
        output_dir (str): Directory to save processed batches.
        task (str): Task type, either "ner" or "sa".
    """
    if task == "ner":
        tokens = dataset_split["tokens"]
        labels = dataset_split["ner_tags"]
        num_samples = len(tokens)
    elif task == "sa":
        tokens = dataset_split["sentence"]
        labels = dataset_split["label"]
        num_samples = len(tokens)
    else:
        raise ValueError("Unknown task: must be 'ner' or 'sa'.")

    for i in tqdm(
        range(0, num_samples, batch_size),
        desc=f"Processing {dataset_name} ({task}) batches",
    ):
        batch_tokens = tokens[i: i + batch_size]
        batch_labels = labels[i: i + batch_size]

        embeddings, padded_labels = process_batch(
            batch_tokens, batch_labels, tokenizer, max_length, model, task=task
        )
        save_batch(
            dataset_name,
            i // batch_size,
            embeddings,
            padded_labels,
            output_dir,
        )


def process_batch(
    batch_inputs: List[Union[List[str], int]],
    batch_labels: List[Union[List[str], int]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    model: torch.nn.Module,
    task: str = "ner",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenizes and encodes a batch, computes embeddings, and pads labels.

    Args:
        batch_inputs: List of token lists (NER) or sentences (SA).
        batch_labels: List of label lists (NER) or ints (SA).
        tokenizer: Tokenizer for the model.
        max_length: Maximum sequence length.
        model: Model to produce embeddings.
        task: Task type, either "ner" or "sa".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Embeddings and padded labels.
    """
    if task == "ner":
        encoded_batch = tokenizer(
            batch_inputs,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # [batch, seq_len, emb_dim]

        # Pad labels to max_length with -1 (ignore index)
        label_tensors = [torch.tensor(lbls) for lbls in batch_labels]
        padded_labels = torch.full((len(label_tensors), max_length), -1)
        for idx, label_tensor in enumerate(label_tensors):
            length = min(len(label_tensor), max_length)
            padded_labels[idx, :length] = label_tensor[:length]

        return embeddings, padded_labels

    elif task == "sa":
        encoded_batch = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding for each sentence
            embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, emb_dim]

        labels = torch.tensor(batch_labels)  # [batch]
        return embeddings, labels

    else:
        raise ValueError(f"Unknown task type: {task}")


def save_batch(
    dataset_name: str,
    batch_idx: int,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str,
) -> None:
    """
    Saves a batch of embeddings and labels to disk as a .pt file.

    Args:
        dataset_name (str): Name of the dataset split.
        batch_idx (int): Batch index.
        embeddings (torch.Tensor): Embedding tensor.
        labels (torch.Tensor): Label tensor.
        output_dir (str): Directory to save the batch.
    """
    save_path = os.path.join(output_dir, dataset_name, f"batch_{batch_idx}.pt")
    data = {
        "embeddings": embeddings.cpu(),
        "labels": labels.cpu(),
    }
    torch.save(data, save_path)


def get_max_length(dataset_split: Dict[str, List[List[str]]]) -> int:
    """
    Returns the maximum sequence length in a dataset split (used for padding).
    Args:
        dataset_split (dict): Dictionary containing a "tokens" key with lists of tokens.
    Returns:
        int: Maximum length found.
    """
    max_length = max(len(tokens) for tokens in dataset_split["tokens"])
    return max_length


class NERAccuracy:
    """
    Computes accuracy for NER tasks, ignoring padding tokens.
    """

    def __init__(self, ignore_index: int = -1) -> None:
        self.correct: int = 0
        self.total: int = 0
        self.ignore_index: int = ignore_index

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        # If predictions are logits, get the argmax
        if predictions.dim() == 3:
            predictions = predictions.argmax(dim=-1)
        assert (
            predictions.shape == labels.shape
        ), f"Shape mismatch: predictions {predictions.shape}, labels {labels.shape}"

        mask = labels != self.ignore_index
        self.correct += int((predictions[mask] == labels[mask]).sum().item())
        self.total += int(mask.sum().item())

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class ClassificationAccuracy:
    """
    Computes accuracy for classification tasks (e.g., Sentiment Analysis).
    """

    def __init__(self) -> None:
        self.correct: int = 0
        self.total: int = 0

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        preds = predictions.argmax(dim=-1)
        self.correct += int((preds == labels).sum().item())
        self.total += labels.size(0)

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class EarlyStopping:
    """
    Implements early stopping for training loops.
    Stops training if validation metric does not improve after a given patience.
    """

    def __init__(
        self,
        patience: int = 10,
        delta: float = 0.0,
        verbose: bool = False,
        mode: str = "min",
    ) -> None:
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
    Saves a PyTorch model as a TorchScript file in the 'models' directory.

    Args:
        model (torch.nn.Module): The model to save.
        name (str): Name for the saved model file.
    """
    if not os.path.isdir("models"):
        os.makedirs("models")

    device = torch.device("cpu")
    model = model.to(device).eval()
    input_dim = model.encoder.input_size
    example_input = torch.randn(
        1, 4, input_dim, device=device
    )  # (batch, seq_len, input_dim)

    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(f"models/{name}.pt")
    print(f"Model saved as models/{name}.pt on device {device}")


def load_model(
    name: str, device: torch.device = torch.device("cpu")
) -> RecursiveScriptModule:
    """
    Loads a TorchScript model from the 'models' directory.

    Args:
        name (str): Name of the model file (without extension).
        device (torch.device): Device to load the model onto.

    Returns:
        RecursiveScriptModule: The loaded model.
    """
    model = torch.jit.load(f"models/{name}.pt", map_location="cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded from models/{name}.pt on device {device}")
    return model


def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The random seed.
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
