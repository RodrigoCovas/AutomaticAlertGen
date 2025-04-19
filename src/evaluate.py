# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule

# other libraries
from typing import Final

# own modules
from src.data import load_data
from src.utils import set_seed
from src.train_functions import t_step

# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main(name: str) -> float:
    """
    This function is the main program.
    """

    _, _, test_data = load_data()
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt").to(device)
    metrics: float = t_step(model, test_data, device)
    return metrics


if __name__ == "__main__":
    print(f"metrics: {main('best_model')}")
