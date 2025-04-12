# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final

# own modules
from src.data import load_data
from src.models import MyMultiTaskModel
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
    batch_size: int = 32
    hidden_size: int = 4
    patience: int = 20
    num_layers: int = 1

    open("nohup.out", "w").close()
    train_data: DataLoader
    val_data: DataLoader

    train_data, val_data, _ = load_data(batch_size)
    
    for batch in range(len(train_data)):
        batch = next(iter(train_data))
        ner_tag_size = batch["labels"].shape[1]
        break

    name: str = f"model_lr_{lr}_hs_{hidden_size}_{batch_size}_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    model: torch.nn.Module = MyMultiTaskModel(hidden_size, num_layers, bidirectional=True, ner_tag_size=ner_tag_size).to(
        device
    )
    ner_loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    sentiment_loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5
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
