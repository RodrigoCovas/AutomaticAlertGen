import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from src.utils import NERAccuracy, ClassificationAccuracy


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    ner_loss_fn: Optional[torch.nn.Module],
    sentiment_loss_fn: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    task: str = "ner",
) -> None:
    ner_acc = NERAccuracy()
    sentiment_acc = ClassificationAccuracy()
    total_loss = 0.0
    num_batches = 0
    model.train()
    for i, batch in enumerate(train_data):
        inputs = batch["embeddings"].to(torch.float32).to(device)
        optimizer.zero_grad()

        if task == "ner":
            ner_tags = batch["labels"].long().to(device)
            ner_predictions, _ = model(inputs)
            loss = ner_loss_fn(ner_predictions.view(-1, ner_predictions.shape[-1]), ner_tags.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            ner_acc.update(ner_predictions, ner_tags)
        elif task == "sa":
            sentiments = batch["labels"].long().to(device)
            _, sentiment_predictions = model(inputs)
            loss = sentiment_loss_fn(sentiment_predictions, sentiments)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            sentiment_acc.update(sentiment_predictions, sentiments)
        else:
            raise ValueError(f"Unknown task: {task}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar(f"{task}/train_loss", avg_loss, epoch)
    if task == "ner":
        writer.add_scalar("train/ner_acc", ner_acc.compute(), epoch)
        ner_acc.reset()
    elif task == "sa":
        writer.add_scalar("train/sentiment_acc", sentiment_acc.compute(), epoch)
        sentiment_acc.reset()


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    ner_loss_fn: Optional[torch.nn.Module],
    sentiment_loss_fn: Optional[torch.nn.Module],
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    task: str = "ner",  # NEW: task argument
) -> float:
    ner_acc = NERAccuracy()
    sentiment_acc = ClassificationAccuracy()
    total_loss = 0.0
    num_batches = 0
    model.eval()
    for i, batch in enumerate(val_data):
        inputs = batch["embeddings"].to(torch.float32).to(device)

        if task == "ner":
            ner_tags = batch["labels"].long().to(device)
            ner_predictions, _ = model(inputs)
            loss = ner_loss_fn(ner_predictions.view(-1, ner_predictions.shape[-1]), ner_tags.view(-1))
            total_loss += loss.item()
            num_batches += 1
            ner_acc.update(ner_predictions, ner_tags)
        elif task == "sa":
            sentiments = batch["labels"].long().to(device)
            _, sentiment_predictions = model(inputs)
            loss = sentiment_loss_fn(sentiment_predictions, sentiments)
            total_loss += loss.item()
            num_batches += 1
            sentiment_acc.update(sentiment_predictions, sentiments)
        else:
            raise ValueError(f"Unknown task: {task}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar(f"{task}/val_loss", avg_loss, epoch)
    if task == "ner":
        writer.add_scalar("val/ner_acc", ner_acc.compute(), epoch)
        ner_acc.reset()
    elif task == "sa":
        writer.add_scalar("val/sentiment_acc", sentiment_acc.compute(), epoch)
        sentiment_acc.reset()
    if scheduler:
        scheduler.step(avg_loss)
    return avg_loss


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
    task: str = "ner",
) -> dict:
    ner_acc = NERAccuracy()
    sentiment_acc = ClassificationAccuracy()
    model.eval()
    for i, batch in enumerate(test_data):
        inputs = batch["embeddings"].to(torch.float32).to(device)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)  # Now [batch, 1, emb_dim]
        if task == "ner":
            ner_tags = batch["labels"].long().to(device)
            ner_predictions, _ = model(inputs)
            ner_acc.update(ner_predictions, ner_tags)
        elif task == "sa":
            sentiments = batch["labels"].long().to(device)
            _, sentiment_predictions = model(inputs)
            sentiment_acc.update(sentiment_predictions, sentiments)
        else:
            raise ValueError(f"Unknown task: {task}")
    return {
        "ner_acc": ner_acc.compute() if task == "ner" else None,
        "sentiment_acc": sentiment_acc.compute() if task == "sa" else None,
    }

