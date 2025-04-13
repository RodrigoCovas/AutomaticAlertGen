import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from src.utils import MAE, EarlyStopping


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    ner_loss_fn: torch.nn.Module,
    sentiment_loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    Train the multi-task model.

    Args:
        model: Multi-task model to train.
        train_data: Dataloader of training data.
        ner_loss_fn: Loss function for NER task.
        sentiment_loss_fn: Loss function for Sentiment Analysis task.
        optimizer: Optimizer.
        writer: Writer for TensorBoard.
        epoch: Current epoch number.
        device: Device for running operations (e.g., 'cuda' or 'cpu').
    """

    ner_mae = MAE()
    sentiment_mae = MAE()
    total_loss = 0.0
    num_batches = 0
    model.train()
    # for batch in train_data:
    for i, batch in enumerate(train_data):
        inputs = batch["embeddings"]
        ner_tags = batch["labels"]
        sentiments = batch["sentiments"]
        inputs = inputs.to(torch.float32).to(device)
        ner_tags = ner_tags.to(torch.float32).to(device)  # NER targets (multi-class)
        sentiments = sentiments.to(torch.float32).to(
            device
        )  # Sentiment targets (binary)
        optimizer.zero_grad()
        ner_predictions, sentiment_predictions = model(inputs)
        print(
            f"NER Predictions: {ner_predictions}, Sentiment Predictions: {sentiment_predictions}, Shapes: {ner_predictions.shape}, {sentiment_predictions.shape}"
        )
        batch_loss = combined_loss(
            ner_predictions,
            ner_tags,
            sentiment_predictions,
            sentiments,
            ner_loss_fn,
            sentiment_loss_fn,
        )
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
        num_batches += 1
        # Update metrics
        ner_mae.update(ner_predictions, ner_tags)
        sentiment_mae.update(sentiment_predictions.squeeze(), sentiments)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/ner_mae", ner_mae.compute(), epoch)
    writer.add_scalar("train/sentiment_mae", sentiment_mae.compute(), epoch)
    ner_mae.reset()
    sentiment_mae.reset()


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    ner_loss_fn: torch.nn.Module,
    sentiment_loss_fn: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> float:
    """
    Validate the multi-task model.

    Args:
        model: Multi-task model to validate.
        val_data: Dataloader of validation data.
        ner_loss_fn: Loss function for NER task.
        sentiment_loss_fn: Loss function for Sentiment Analysis task.
        scheduler: Learning rate scheduler.
        writer: Writer for TensorBoard.
        epoch: Current epoch number.
        device: Device for running operations (e.g., 'cuda' or 'cpu').

    Returns:
        Average validation loss (float).
    """

    ner_mae = MAE()
    sentiment_mae = MAE()
    total_loss = 0.0
    num_batches = 0
    model.eval()
    for i, batch in enumerate(val_data):
        inputs = batch["embeddings"]
        ner_tags = batch["labels"]
        sentiments = batch["sentiments"]
        inputs = inputs.to(torch.float32).to(device)
        ner_tags = ner_tags.to(torch.float32).to(device)
        sentiments = sentiments.to(torch.float32).to(device)
        ner_predictions, sentiment_predictions = model(inputs)
        print(
            f"NER Predictions: {ner_predictions}, Sentiment Predictions: {sentiment_predictions}, Shapes: {ner_predictions.shape}, {sentiment_predictions.shape}"
        )
        batch_loss = combined_loss(
            ner_predictions,
            ner_tags,
            sentiment_predictions,
            sentiments,
            ner_loss_fn,
            sentiment_loss_fn,
        )
        total_loss += batch_loss.item()
        num_batches += 1
        # Update metrics
        ner_mae.update(ner_predictions, ner_tags)
        sentiment_mae.update(sentiment_predictions.squeeze(), sentiments)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/ner_mae", ner_mae.compute(), epoch)
    writer.add_scalar("val/sentiment_mae", sentiment_mae.compute(), epoch)
    ner_mae.reset()
    sentiment_mae.reset()
    if scheduler:
        scheduler.step(avg_loss)
    return avg_loss


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> dict:
    """
    Test the multi-task model.

    Args:
        model: Multi-task model to test.
        test_data: Dataloader of testing data.
        device: Device for running operations (e.g., 'cuda' or 'cpu').

    Returns:
        A dictionary containing metrics for NER and Sentiment Analysis tasks.
    """

    ner_mae = MAE()
    sentiment_mae = MAE()
    model.eval()
    for i, batch in enumerate(test_data):
        inputs = batch["embeddings"]
        ner_tags = batch["labels"]
        sentiments = batch["sentiments"]
        inputs = inputs.to(torch.float32).to(device)
        ner_tags = ner_tags.to(torch.float32).to(device)
        sentiments = sentiments.to(torch.float32).to(device)
        ner_predictions, sentiment_predictions = model(inputs)
        # Update metrics
        ner_mae.update(ner_predictions, ner_tags)
        sentiment_mae.update(sentiment_predictions.squeeze(), sentiments)
    return {
        "ner_mae": ner_mae.compute(),
        "sentiment_mae": sentiment_mae.compute(),
    }


def combined_loss(
    ner_predictions,
    ner_targets,
    sentiment_predictions,
    sentiment_targets,
    ner_loss_fn,
    sentiment_loss_fn,
):
    """
    Compute the combined loss for multi-task learning.

    Args:
        ner_predictions: Predicted NER tags. Shape: [batch, ner_tag_size].
        ner_targets: True NER tags. Shape: [batch].
        sentiment_predictions: Predicted sentiment scores. Shape: [batch, 1].
        sentiment_targets: True sentiment labels. Shape: [batch].
        ner_loss_fn: Loss function for NER task (e.g., CrossEntropyLoss).
        sentiment_loss_fn: Loss function for sentiment task (e.g., BCEWithLogitsLoss).

    Returns:
        Combined loss value.
    """
    loss_ner = ner_loss_fn(ner_predictions, ner_targets)
    loss_sentiment = sentiment_loss_fn(
        sentiment_predictions.squeeze(), sentiment_targets.float()
    )
    print(f"NER Loss: {loss_ner.item()}, Sentiment Loss: {loss_sentiment.item()}")
    return loss_ner + loss_sentiment
