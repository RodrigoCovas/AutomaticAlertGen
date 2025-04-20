from tqdm import tqdm
import torch
import os
import random
import numpy as np
from torch.jit import RecursiveScriptModule
from typing import Optional

def compute_class_weights(train_loader, num_classes):
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    for batch in train_loader:
        labels = batch["labels"].view(-1)
        mask = labels != -1  # Exclude padding
        labels = labels[mask]
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum()
    # Avoid division by zero
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    class_weights = class_weights * (num_classes / class_weights.sum())  # Normalize
    return class_weights


def process_in_batches(
    dataset_split,
    dataset_name,
    batch_size,
    tokenizer,
    max_length,
    model,
    output_dir,
    task="ner",
):
    """
    Processes the dataset in batches for either NER or SA.

    Args:
        dataset_split: HuggingFace Dataset split or dict with data.
        dataset_name: str, name of the split ("train", "validation", "test").
        batch_size: int
        tokenizer: HuggingFace tokenizer
        max_length: int
        model: HuggingFace model
        output_dir: str, where to save batches
        task: "ner" or "sa"
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
        batch_tokens = tokens[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
            
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

def process_batch(batch_inputs, batch_labels, tokenizer, max_length, model, task="ner"):
    """
    Unified batch processor for NER and SA.
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

        # Pad labels to max_length
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
            # Extract only the [CLS] token embedding for each sentence
            embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, emb_dim]

        labels = torch.tensor(batch_labels)  # [batch]
        return embeddings, labels

    else:
        raise ValueError(f"Unknown task type: {task}")


def save_batch(dataset_name, batch_idx, embeddings, labels, output_dir):
    """
    Saves processed batch embeddings and labels to disk.
    For SA, 'sentiments' can be None.
    """
    save_path = os.path.join(output_dir, dataset_name, f"batch_{batch_idx}.pt")
    data = {
        "embeddings": embeddings.cpu(),
        "labels": labels.cpu(),
    }
    torch.save(data, save_path)


def get_max_length(dataset_split):
    """Returns the maximum length of tokens in the dataset split."""
    max_length = max(len(tokens) for tokens in dataset_split["tokens"])
    return max_length

class NERAccuracy:
    """
    Accuracy metric for NER tasks, measuring token-level accuracy.

    Attributes:
        correct: number of correct predictions.
        total: number of total valid tokens (excluding padding/masked tokens).
    """

    def __init__(self, ignore_index: int = -1) -> None:
        self.correct = 0
        self.total = 0
        self.ignore_index = ignore_index

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Updates the number of correct predictions and total valid tokens.

        Args:
            predictions: [batch_size, seq_len, num_classes] or [batch_size, seq_len]
            labels: [batch_size, seq_len]
        """
        # Convertir logits a etiquetas si es necesario
        if predictions.dim() == 3:
            predictions = predictions.argmax(dim=-1)

        assert (
            predictions.shape == labels.shape
        ), f"Shape mismatch: predictions {predictions.shape}, labels {labels.shape}"

        mask = labels != self.ignore_index
        self.correct += (predictions[mask] == labels[mask]).sum().item()
        self.total += mask.sum().item()

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class ClassificationAccuracy:
    """
    Accuracy for classification tasks like Sentiment Analysis.

    Attributes:
        correct: number of correct predictions.
        total: total number of examples.
    """

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Args:
            predictions: logits or probabilities, shape [batch_size, num_classes]
            labels: ground truth labels, shape [batch_size]
        """
        preds = predictions.argmax(dim=-1)
        self.correct += (preds == labels).sum().item()
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
    Save the model as a TorchScript traced model on CPU (for maximum portability).

    Args:
        model: PyTorch model to save.
        name: Filename (without extension) to save the model under 'models/' folder.
    """
    if not os.path.isdir("models"):
        os.makedirs("models")

    device = torch.device("cpu")
    model = model.to(device).eval()
    input_dim = model.encoder.input_size
    example_input = torch.randn(1, 4, input_dim, device=device)  # (batch, seq_len, input_dim)

    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(f"models/{name}.pt")
    print(f"Model saved as models/{name}.pt on device {device}")

def load_model(name: str, device: torch.device = torch.device("cpu")) -> RecursiveScriptModule:
    """
    Load a TorchScript model from the 'models' folder and move it to the specified device.

    Args:
        name: Filename (without extension) of the model to load.
        device: Device to move the model to.

    Returns:
        Loaded TorchScript model on the specified device.
    """
    model = torch.jit.load(f"models/{name}.pt", map_location="cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded from models/{name}.pt on device {device}")
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
