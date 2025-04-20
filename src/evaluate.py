import torch
from typing import Final
from src.data import load_data
from src.utils import set_seed, load_model
from src.train_functions import t_step

# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

def main(name: str):
    """
    Evaluate the sequential model on both NER and SA test sets.
    """
    (
        _,
        _,
        test_loader_ner,
        _,
        _,
        test_loader_sa,
    ) = load_data()

    # Load the model using the new load_model function
    model = load_model(name, device)

    # Evaluate NER
    ner_metrics = t_step(model, test_loader_ner, device, task="ner")
    print(f"NER Test Accuracy: {ner_metrics['ner_acc']:.4f}")

    # Evaluate Sentiment Analysis
    sa_metrics = t_step(model, test_loader_sa, device, task="sa")
    print(f"Sentiment Analysis Test Accuracy: {sa_metrics['sentiment_acc']:.4f}")

    return {"ner": ner_metrics, "sa": sa_metrics}

if __name__ == "__main__":
    metrics = main('best_model')