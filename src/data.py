import os
from datasets import load_dataset
from typing import List, Tuple, Dict, Optional
from datasets import DatasetDict, Dataset as HFDataset
from transformers import BertTokenizer, BertModel
from src.utils import (
    process_in_batches,
    get_max_length,
)
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader


class NER_Dataset(TorchDataset):
    def __init__(
        self, directory: str = "./data/ner/train", pad_batches: bool = True
    ) -> None:
        """
        Custom Dataset class for Named Entity Recognition (NER) tasks using
        precomputed embeddings.

        Args:
            directory (str): Directory containing the batch files (torch tensors).
            pad_batches (bool): If True, smaller batches will be padded to match the
            expected batch size.
        """
        super().__init__()

        self._file_paths: List[str] = [
            os.path.join(directory, file) for file in sorted(os.listdir(directory))
        ]

        if not self._file_paths:
            raise ValueError(f"No files found in directory: {directory}")

        first_batch: Dict[str, torch.Tensor] = torch.load(
            self._file_paths[0], weights_only=True
        )
        self._batch_size: int = first_batch["embeddings"].shape[0]

        self._batch_in_memory: Tuple[int, Dict[str, torch.Tensor] | None] = (-1, None)
        self.pad_batches: bool = pad_batches

    def __len__(self) -> int:
        total_samples: int = 0
        for file_path in self._file_paths:
            batch_data: Dict[str, torch.Tensor] = torch.load(
                file_path, weights_only=True
            )
            total_samples += batch_data["embeddings"].shape[0]
        return total_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        batch_idx: int = index // self._batch_size
        rel_idx: int = index % self._batch_size

        if batch_idx < 0 or batch_idx >= len(self._file_paths):
            raise IndexError(f"Index {index} is out of range for dataset.")

        if self._batch_in_memory[0] != batch_idx:
            self._cache_batch(batch_idx)

        batch_data: Optional[Dict[str, torch.Tensor]] = self._batch_in_memory[1]

        if batch_data is not None:
            actual_batch_size: int = batch_data["embeddings"].shape[0]
        else:
            # handle the None case appropriately
            raise ValueError("batch_data is None")

        if rel_idx >= actual_batch_size:
            raise IndexError(
                f"Relative index {rel_idx} exceeds actual batch size {actual_batch_size}"
            )

        if batch_data is None:
            # handle the None case appropriately
            raise ValueError("batch_data is None")
        return {field: batch_data[field][rel_idx] for field in ("embeddings", "labels")}

    def _cache_batch(self, n_batch: int) -> None:
        if n_batch < 0 or n_batch >= len(self._file_paths):
            raise IndexError(f"Batch index {n_batch} is out of range for file paths.")

        raw_batch: Dict[str, torch.Tensor] = torch.load(
            self._file_paths[n_batch], weights_only=True
        )
        actual_batch_size: int = raw_batch["embeddings"].shape[0]

        if actual_batch_size < self._batch_size:
            if self.pad_batches:
                raw_batch = self._pad_batch(raw_batch, target_size=self._batch_size)
            else:
                print(
                    f"""Warning: Skipping smaller batch {n_batch} with
                    size {actual_batch_size}"""
                )
                raise ValueError(
                    f"Batch {n_batch} has inconsistent size: {actual_batch_size}"
                )

        self._batch_in_memory = (n_batch, raw_batch)

    def _pad_batch(
        self, batch_data: Dict[str, torch.Tensor], target_size: int
    ) -> Dict[str, torch.Tensor]:
        padded_data: Dict[str, torch.Tensor] = {}

        for field in ("embeddings", "labels"):
            original_data: torch.Tensor = batch_data[field]

            padding_shape: List[int] = [target_size - original_data.shape[0]] + list(
                original_data.shape[1:]
            )

            padding: torch.Tensor = torch.zeros(
                *padding_shape, dtype=original_data.dtype
            ).to(original_data.device)

            padded_data[field] = torch.cat([original_data, padding], dim=0)

        return padded_data


class SA_Dataset(TorchDataset):
    def __init__(
        self, directory: str = "./data/sa/train", pad_batches: bool = True
    ) -> None:
        """
        Custom Dataset class for Sentiment Analysis (SA) tasks using precomputed
        embeddings.

        Args:
            directory (str): Directory containing the batch files (torch tensors).
            pad_batches (bool): If True, smaller batches will be padded to match the
            expected batch size.
        """
        super().__init__()
        # Collect all file paths from the given directory, sorted for consistent ordering
        self._file_paths: List[str] = [
            os.path.join(directory, file) for file in sorted(os.listdir(directory))
        ]
        # Ensure that the directory contains at least one file
        if not self._file_paths:
            raise ValueError(f"No files found in directory: {directory}")

        # Determine the expected batch size from the first batch file
        first_batch: Dict[str, torch.Tensor] = torch.load(
            self._file_paths[0], weights_only=True
        )
        self._batch_size: int = first_batch["embeddings"].shape[0]

        # Tuple to cache (batch_index, batch_data) in memory for efficient access
        self._batch_in_memory: Tuple[int, Dict[str, torch.Tensor] | None] = (-1, None)
        self.pad_batches: bool = pad_batches

    def __len__(self) -> int:
        """
        Returns the total number of samples across all batches.
        """
        total_samples: int = 0
        for file_path in self._file_paths:
            batch_data: Dict[str, torch.Tensor] = torch.load(
                file_path, weights_only=True
            )
            total_samples += batch_data["embeddings"].shape[0]
        return total_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single sample by global index across all batches.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing 'embeddings' and 'labels'
            tensors.
        """
        # Determine batch index and relative index within the batch
        batch_idx: int = index // self._batch_size
        rel_idx: int = index % self._batch_size

        # Validate batch index
        if batch_idx < 0 or batch_idx >= len(self._file_paths):
            raise IndexError(f"Index {index} is out of range for dataset.")

        # Cache the batch if it's not already in memory
        if self._batch_in_memory[0] != batch_idx:
            self._cache_batch(batch_idx)

        # Retrieve the cached batch data
        batch_data: Optional[Dict[str, torch.Tensor]] = self._batch_in_memory[1]
        if batch_data is not None:
            actual_batch_size: int = batch_data["embeddings"].shape[0]
        else:
            # handle the None case appropriately
            raise ValueError("batch_data is None")

        if rel_idx >= actual_batch_size:
            raise IndexError(
                f"Relative index {rel_idx} exceeds actual batch size {actual_batch_size}"
            )

        # Return the sample (embedding and label) at the relative index
        return {field: batch_data[field][rel_idx] for field in ("embeddings", "labels")}

    def _cache_batch(self, n_batch: int) -> None:
        """
        Loads and caches a specific batch in memory.

        Args:
            n_batch (int): Batch index to cache.

        Raises:
            IndexError: If the batch index is out of range.
            ValueError: If a smaller batch is encountered and cannot be padded.
        """
        # Validate batch index before loading
        if n_batch < 0 or n_batch >= len(self._file_paths):
            raise IndexError(f"Batch index {n_batch} is out of range for file paths.")

        # Load the batch data from file
        raw_batch: Dict[str, torch.Tensor] = torch.load(
            self._file_paths[n_batch], weights_only=True
        )
        actual_batch_size: int = raw_batch["embeddings"].shape[0]

        # If the batch is smaller than the expected size, pad it if allowed
        if actual_batch_size < self._batch_size:
            if self.pad_batches:
                raw_batch = self._pad_batch(raw_batch, target_size=self._batch_size)
            else:
                print(
                    f"""Warning: Skipping smaller batch {n_batch}
                    with size {actual_batch_size}"""
                )
                raise ValueError(
                    f"Batch {n_batch} has inconsistent size: {actual_batch_size}"
                )

        # Store the batch in memory for fast access
        self._batch_in_memory = (n_batch, raw_batch)

    def _pad_batch(
        self, batch_data: Dict[str, torch.Tensor], target_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        Pads a smaller batch to the target size by adding zero tensors.

        Args:
            batch_data (Dict[str, torch.Tensor]): The original batch data.
            target_size (int): The desired batch size.

        Returns:
            Dict[str, torch.Tensor]: The padded batch data.
        """
        padded_data: Dict[str, torch.Tensor] = {}
        for field in ("embeddings", "labels"):
            original_data: torch.Tensor = batch_data[field]
            # Calculate the shape of the padding tensor
            padding_shape: List[int] = [target_size - original_data.shape[0]] + list(
                original_data.shape[1:]
            )
            # Create a tensor of zeros for padding
            padding: torch.Tensor = torch.zeros(
                *padding_shape, dtype=original_data.dtype
            ).to(original_data.device)
            # Concatenate the original data with the padding tensor along
            # the batch dimension
            padded_data[field] = torch.cat([original_data, padding], dim=0)
        return padded_data


def load_data(
    batch_size: int = 32,
) -> Tuple[
    DataLoader[Dict[str, torch.Tensor]],
    DataLoader[Dict[str, torch.Tensor]],
    DataLoader[Dict[str, torch.Tensor]],
    DataLoader[Dict[str, torch.Tensor]],
    DataLoader[Dict[str, torch.Tensor]],
    DataLoader[Dict[str, torch.Tensor]],
]:
    """
    Loads (and downloads if necessary) the data for NER and Sentiment Analysis tasks,
    returning DataLoaders for train, validation, and test splits for both tasks.

    Args:
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        Tuple of DataLoaders:
            (train_loader_ner, val_loader_ner, test_loader_ner,
             train_loader_sa, val_loader_sa, test_loader_sa)
    """
    # Check if data directories exist, otherwise download and preprocess data
    if not os.path.exists("./data/ner") or not os.path.exists("./data/sa"):
        download_data(batch_size)

    # Instantiate custom datasets for NER and SA
    train_ner: NER_Dataset = NER_Dataset(directory="./data/ner/train")
    val_ner: NER_Dataset = NER_Dataset(directory="./data/ner/validation")
    test_ner: NER_Dataset = NER_Dataset(directory="./data/ner/test")

    train_sa: SA_Dataset = SA_Dataset(directory="./data/sa/train")
    val_sa: SA_Dataset = SA_Dataset(directory="./data/sa/validation")
    test_sa: SA_Dataset = SA_Dataset(directory="./data/sa/test")

    # Create DataLoaders for NER
    train_loader_ner: DataLoader[Dict[str, torch.Tensor]] = DataLoader(
        train_ner, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader_ner: DataLoader[Dict[str, torch.Tensor]] = DataLoader(
        val_ner, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader_ner: DataLoader[Dict[str, torch.Tensor]] = DataLoader(
        test_ner, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # Create DataLoaders for Sentiment Analysis
    train_loader_sa: DataLoader[Dict[str, torch.Tensor]] = DataLoader(
        train_sa, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader_sa: DataLoader[Dict[str, torch.Tensor]] = DataLoader(
        val_sa, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader_sa: DataLoader[Dict[str, torch.Tensor]] = DataLoader(
        test_sa, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # Return all DataLoaders as a tuple
    return (
        train_loader_ner,
        val_loader_ner,
        test_loader_ner,
        train_loader_sa,
        val_loader_sa,
        test_loader_sa,
    )


def download_data(batch_size: int = 32) -> None:
    """
    Downloads and preprocesses the datasets for NER (CoNLL2003) and
    Sentiment Analysis (Financial Phrasebank).
    Saves the processed data as batches of embeddings and labels.

    Args:
        batch_size (int): The batch size to use for processing and saving.
    """
    # Download CoNLL2003 dataset for NER
    conll_dataset: DatasetDict = load_dataset("conll2003", trust_remote_code=True)

    # Load BERT tokenizer and model for encoding the data
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model: BertModel = BertModel.from_pretrained("bert-base-uncased")

    # Determine the maximum sequence length across all splits for padding
    length_list: List[int] = []
    for dataset_name in ["train", "validation", "test"]:
        length_list.append(get_max_length(conll_dataset[dataset_name]))
    max_length: int = max(length_list) + 2  # +2 for special tokens
    print(f"Max length of tokens: {max_length}")

    # Create output directories for NER data
    output_dir: str = "./data/ner"
    datasets_names: List[str] = ["train", "validation", "test"]
    for dataset_name in datasets_names:
        os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)

    # Process and save NER data in batches
    for dataset_name in datasets_names:
        process_in_batches(
            conll_dataset[dataset_name],
            dataset_name,
            batch_size,
            tokenizer,
            max_length,
            model,
            output_dir,
            task="ner",
        )

    # Download Financial Phrasebank dataset for Sentiment Analysis
    phrasebank: DatasetDict = load_dataset(
        "financial_phrasebank", "sentences_66agree", trust_remote_code=True
    )

    full_dataset: HFDataset = phrasebank["train"]

    # Split dataset into train, validation, and test
    train_testvalid: DatasetDict = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset: HFDataset = train_testvalid["train"]
    testvalid_dataset: HFDataset = train_testvalid["test"]

    test_valid: DatasetDict = testvalid_dataset.train_test_split(test_size=0.5, seed=42)
    validation_dataset: HFDataset = test_valid["train"]
    test_dataset: HFDataset = test_valid["test"]

    splits: Dict[str, HFDataset] = {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }

    # Create output directories for SA data
    sa_output_dir: str = "./data/sa"
    for split_name in splits:
        os.makedirs(os.path.join(sa_output_dir, split_name), exist_ok=True)

    # Process and save SA data in batches
    for split_name, split_data in splits.items():
        process_in_batches(
            split_data,
            split_name,
            batch_size,
            tokenizer,
            max_length,
            model,
            sa_output_dir,
            task="sa",
        )


if __name__ == "__main__":
    # Load all DataLoaders for NER and SA tasks
    (
        train_loader_ner,
        val_loader_ner,
        test_loader_ner,
        train_loader_sa,
        val_loader_sa,
        test_loader_sa,
    ) = load_data(batch_size=32)

    # Print the shape of the first batch for NER
    for _ in range(len(train_loader_ner)):
        batch: Dict[str, torch.Tensor] = next(iter(train_loader_ner))
        print(batch["embeddings"].shape, batch["labels"].shape)
        break

    # Print the shape of the first batch for SA
    for _ in range(len(train_loader_sa)):
        batch = next(iter(train_loader_sa))
        print(batch["embeddings"].shape, batch["labels"].shape)
        break
