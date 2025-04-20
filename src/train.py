# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final
import os

# own modules
from src.data import load_data
from src.models import CombinedModel
from src.train_functions import train_step, val_step
from src.utils import set_seed, save_model, EarlyStopping, compute_class_weights

# static variables
DATA_PATH: Final[str] = "data"

# set device and seed
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")
set_seed(42)


def main() -> None:
    """
    Main program for sequential NER and SA training.
    """
    # Hyperparameters
    epochs_ner: int = 1
    epochs_sa: int = 1
    lr_ner: float = 1e-3
    lr_sa: float = 1e-3
    batch_size: int = 64
    hidden_size: int = 512
    patience: int = 10
    num_layers: int = 2
    bidirectional: bool = True
    num_ner_classes: int = 10
    num_sa_classes: int = 3

    open("nohup.out", "w").close()

    (
        train_loader_ner,
        val_loader_ner,
        _,
        train_loader_sa,
        val_loader_sa,
        _
    ) = load_data(batch_size=batch_size)

    # Get model input/output size from a sample batch
    for batch in train_loader_ner:
        input_dim = batch["embeddings"].shape[2]  # Size of embedding vector
        break

    name: str = f"model_hs_{hidden_size}_{batch_size}_{epochs_ner}_{epochs_sa}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    model: torch.nn.Module = CombinedModel(
        input_dim=input_dim,
        hidden_dim=hidden_size,
        num_layers=num_layers,
        num_ner_classes=num_ner_classes,
        sentiment_classes=num_sa_classes,
        bidirectional=bidirectional,
    ).to(device)

    ner_class_weights = compute_class_weights(train_loader_ner, num_ner_classes).to(device)
    sa_class_weights = compute_class_weights(train_loader_sa, num_sa_classes).to(device)

    ner_loss: torch.nn.Module = torch.nn.CrossEntropyLoss(weight=ner_class_weights, ignore_index=-1)
    sentiment_loss: torch.nn.Module = torch.nn.CrossEntropyLoss(weight=sa_class_weights)
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_ner, weight_decay=1e-4
    )
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
    )
    early_stopping_ner: EarlyStopping = EarlyStopping(patience=patience, mode="min", delta=1e-4)

    # ====== 1. Train on NER ======
    print("Training NER...")
    best_val_loss_ner = float('inf')
    best_model_path_ner = f"models/{name}_ner_best.pth"

    for epoch in tqdm(range(epochs_ner)):
        train_step(
            model=model,
            train_data=train_loader_ner,
            ner_loss_fn=ner_loss,
            sentiment_loss_fn=None,
            optimizer=optimizer,
            writer=writer,
            epoch=epoch,
            device=device,
            task="ner"
        )
        val_loss: float = val_step(
            model=model,
            val_data=val_loader_ner,
            ner_loss_fn=ner_loss,
            sentiment_loss_fn=None,
            scheduler=scheduler,
            writer=writer,
            epoch=epoch,
            device=device,
            task="ner"
        )
        # Save best model if validation loss improves
        if val_loss < best_val_loss_ner:
            best_val_loss_ner = val_loss
            torch.save(model.state_dict(), best_model_path_ner)
            print(f"New best NER model saved at epoch {epoch+1} with val_loss {val_loss:.4f}")

        early_stopping_ner(val_loss)
        if early_stopping_ner.early_stop:
            print("Early stopping triggered (NER).")
            break
        scheduler.step(val_loss)

    # Load best NER model before moving on
    model.load_state_dict(torch.load(best_model_path_ner))
    print(f"Loaded best NER model from {best_model_path_ner}")

    # Delete the best NER checkpoint file
    if os.path.exists(best_model_path_ner):
        os.remove(best_model_path_ner)
        print(f"Deleted temporary file: {best_model_path_ner}")

    # ====== 2. Freeze encoder ======
    print("Freezing encoder for SA training...")
    for param in model.encoder.parameters():
        param.requires_grad = False

    # ====== 3. Train on SA ======
    optimizer_sa = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_sa, weight_decay=1e-4
    )
    scheduler_sa = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_sa, mode="min", patience=2, factor=0.5
    )
    early_stopping_sa = EarlyStopping(patience=patience, mode="min", delta=1e-4)

    best_val_loss_sa = float('inf')
    best_model_path_sa = f"models/{name}_sa_best.pth"

    print("Training Sentiment Analysis...")
    for epoch in tqdm(range(epochs_sa)):
        train_step(
            model=model,
            train_data=train_loader_sa,
            ner_loss_fn=None,
            sentiment_loss_fn=sentiment_loss,
            optimizer=optimizer_sa,
            writer=writer,
            epoch=epoch,
            device=device,
            task="sa"
        )
        val_loss: float = val_step(
            model=model,
            val_data=val_loader_sa,
            ner_loss_fn=None,
            sentiment_loss_fn=sentiment_loss,
            scheduler=scheduler_sa,
            writer=writer,
            epoch=epoch,
            device=device,
            task="sa"
        )
        # Save best model if validation loss improves
        if val_loss < best_val_loss_sa:
            best_val_loss_sa = val_loss
            torch.save(model.state_dict(), best_model_path_sa)
            print(f"New best SA model saved at epoch {epoch+1} with val_loss {val_loss:.4f}")

        early_stopping_sa(val_loss)
        if early_stopping_sa.early_stop:
            print("Early stopping triggered (SA).")
            break
        scheduler_sa.step(val_loss)

    # Load best SA model before saving as TorchScript
    model.load_state_dict(torch.load(best_model_path_sa))
    print(f"Loaded best SA model from {best_model_path_sa}")

    # Delete the best SA checkpoint file
    if os.path.exists(best_model_path_sa):
        os.remove(best_model_path_sa)
        print(f"Deleted temporary file: {best_model_path_sa}")

    save_model(model, name, device)


if __name__ == "__main__":
    main()  