import torch
from typing import Final, Dict, Optional
from src.data import load_data
from src.utils import set_seed, load_model
from src.train_functions import t_step

# Static configuration variables
DATA_PATH: Final[str] = "data"  # Path to data directory (not used directly here)
NUM_CLASSES: Final[int] = 10  # Number of classes (not used directly here)

# Set device to GPU if available, otherwise CPU
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# Set random seed for reproducibility
set_seed(42)


def main(name: str) -> None:
    """
    Evaluate the sequential model on both NER and Sentiment Analysis (SA) test sets.

    Args:
        name (str): The name (or path) of the model to load.
    """
    # Load data loaders; only test loaders are used here
    (
        _,
        _,
        test_loader_ner,
        _,
        _,
        test_loader_sa,
    ) = load_data()

    # Load the pre-trained model (using a utility function that handles device placement)
    model: torch.nn.Module = load_model(name, device)

    # Evaluate on the NER test set
    ner_metrics: Dict[str, Optional[float]] = t_step(
        model, test_loader_ner, device, task="ner"
    )
    print(f"NER Test Accuracy: {ner_metrics['ner_acc']:.4f}")

    # Evaluate on the Sentiment Analysis test set
    sa_metrics: Dict[str, Optional[float]] = t_step(
        model, test_loader_sa, device, task="sa"
    )
    print(f"Sentiment Analysis Test Accuracy: {sa_metrics['sentiment_acc']:.4f}")


if __name__ == "__main__":
    main("best_model")
