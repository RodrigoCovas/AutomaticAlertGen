# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final

# own modules
from src.data import load_data
from src.models import CombinedModel
from src.train_functions import train_step, val_step
from src.utils import set_seed, save_model, EarlyStopping

# static variables
DATA_PATH: Final[str] = "data"

# set device and seed
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
set_seed(42)


def main() -> None:
    """
    This function is the main program for training.
    """

    # Hyperparameters
    epochs: int = 1
    lr: float = 1e-3
    batch_size: int = 16
    hidden_size: int = 32
    patience: int = 5
    num_layers: int = 2
    bidirectional: bool = True
    num_ner_classes: int = 10
    num_sa_classes: int = 3

    open("nohup.out", "w").close()
    train_data: DataLoader
    val_data: DataLoader

    train_data, val_data, _ = load_data(batch_size)
    
    """for batch in range(len(train_data)):
        batch = next(iter(train_data))
        ner_tag_size = batch["labels"].shape[1]
        break"""

    # Get model input/output size from a sample batch
    for batch in train_data:
        input_dim = batch["embeddings"].shape[2]  # Size of embedding vector
        break

    name: str = f"model_lr_{lr}_hs_{hidden_size}_{batch_size}_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    model: torch.nn.Module = CombinedModel(
        input_dim=input_dim,
        hidden_dim=hidden_size,
        num_layers=num_layers,
        num_ner_classes=num_ner_classes,
        sentiment_classes=num_sa_classes,
        bidirectional=bidirectional,
    ).to(device)
    
    ner_loss: torch.nn.Module = torch.nn.CrossEntropyLoss(ignore_index=-1)
    sentiment_loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
    )
    early_stopping: EarlyStopping = EarlyStopping(patience=patience, mode="min")

    val_losses: list[float] = []
    for epoch in tqdm(range(epochs)):
        train_step(
            model=model,
            train_data=train_data,
            ner_loss_fn=ner_loss,
            sentiment_loss_fn=sentiment_loss,
            optimizer=optimizer,
            writer=writer,
            epoch=epoch,
            device=device,
        )
        val_loss: float = val_step(
            model=model,
            val_data=val_data,
            ner_loss_fn=ner_loss,
            sentiment_loss_fn=sentiment_loss,
            scheduler=scheduler,
            writer=writer,
            epoch=epoch,
            device=device,
        )
        val_losses.append(val_loss)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        scheduler.step(val_loss)
    save_model(model, name)


if __name__ == "__main__":
    main()
