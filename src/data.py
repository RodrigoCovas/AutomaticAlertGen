import os
from datasets import load_dataset
from transformers import pipeline
from transformers import BertTokenizer, BertModel
from src.utils import (
    extract_sentences,
    add_sentiments,
    analyze_with_progress,
    process_in_batches,
    get_max_length,
)
import torch
from torch.utils.data import Dataset, DataLoader


class NER_SA_Dataset(Dataset):
    def __init__(self, directory: str = "./data/train"):
        super().__init__()

        # Load all file paths from the directory
        self._file_paths = [
            os.path.join(directory, file) for file in os.listdir(directory)
        ]

        # Check if there are any files in the directory
        if not self._file_paths:
            raise ValueError(f"No files found in directory: {directory}")

        # Get batch size from the first file
        self._batch_size = torch.load(self._file_paths[0], weights_only=True)[
            "embeddings"
        ].shape[0]
        self._batch_in_memory = (-1, None)

    def __len__(self):
        # Calculate total number of samples across all batches
        last_batch_size = torch.load(self._file_paths[-1], weights_only=True)[
            "embeddings"
        ].shape[0]
        return (len(self._file_paths) - 1) * self._batch_size + last_batch_size


    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        batch_idx = index // self._batch_size

        # Validate batch index
        if batch_idx < 0 or batch_idx >= len(self._file_paths):
            raise IndexError(f"Index {index} is out of range for dataset.")

        # Cache batch if not already cached
        if self._batch_in_memory[0] != batch_idx:
            self._cache_batch(batch_idx)

        rel_idx = index % self._batch_size
        batch_data = self._batch_in_memory[1]
        
        actual_batch_size = batch_data["embeddings"].shape[0]
        if rel_idx >= actual_batch_size:
            raise IndexError(f"Relative index {rel_idx} exceeds actual batch size {actual_batch_size}")

        return {
            field: batch_data[field][rel_idx]
            for field in ("embeddings", "labels", "sentiments")
        }

    def _cache_batch(self, n_batch: int):
        """
        Stores the desired batch in memory so as not to read it each
        "__getitem__" if the desired item is in the batch.
        """
        # Validate batch index before accessing file paths
        if n_batch < 0 or n_batch >= len(self._file_paths):
            raise IndexError(f"Batch index {n_batch} is out of range for file paths.")
        
        self._batch_in_memory = (
            n_batch,
            torch.load(self._file_paths[n_batch], weights_only=True),
        )



def load_data(batch_size: int = 32):
    if not os.path.exists("./data/train"):
        download_data(batch_size)

    # Create datasets for train, validation, and test splits
    train_dataset = NER_SA_Dataset(directory="./data/train")
    validation_dataset = NER_SA_Dataset(directory="./data/validation")
    test_dataset = NER_SA_Dataset(directory="./data/test")

    # Create DataLoaders for each split
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return train_loader, validation_loader, test_loader


def download_data(batch_size: int = 32):
    # Load the original CoNLL-2003 dataset
    dataset = load_dataset("conll2003", trust_remote_code=True)

    train_sentences = extract_sentences(dataset["train"])
    validation_sentences = extract_sentences(dataset["validation"])
    test_sentences = extract_sentences(dataset["test"])

    # Load the sentiment-analysis pipeline
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    train_sentiments = analyze_with_progress(train_sentences, sentiment_analyzer)
    validation_sentiments = analyze_with_progress(
        validation_sentences, sentiment_analyzer
    )
    test_sentiments = analyze_with_progress(test_sentences, sentiment_analyzer)

    dataset["train"] = add_sentiments(dataset["train"], train_sentiments)
    dataset["validation"] = add_sentiments(dataset["validation"], validation_sentiments)
    dataset["test"] = add_sentiments(dataset["test"], test_sentiments)
    
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    length_list = []
    for dataset_name in ["train", "validation", "test"]:
        length_list.append(get_max_length(dataset[dataset_name]))
    max_length = max(length_list) + 2  # Adding 2 for [CLS] and [SEP] tokens
    print(f"Max length of tokens: {max_length}")

    # Directory to save processed batches
    output_dir = "./data"
    datasets_names = ["train", "validation", "test"]
    for dataset_name in datasets_names:
        os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)

    # Process all splits (train/validation/test)
    for dataset_name in datasets_names:
        process_in_batches(
            dataset[dataset_name],
            dataset_name,
            batch_size,
            tokenizer,
            max_length,
            model,
            output_dir,
        )


if __name__ == "__main__":
    train_loader, validation_loader, test_loader = load_data(batch_size=32)
    for batch in range(len(train_loader)):
        batch = next(iter(train_loader))
        print(batch["embeddings"].shape, batch["labels"].shape, batch["sentiments"].shape)
        break
