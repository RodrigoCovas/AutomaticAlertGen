import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from src.utils import MAE, NERAccuracy, ClassificationAccuracy, EarlyStopping


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

    ner_acc = NERAccuracy()
    sentiment_acc = ClassificationAccuracy()
    total_loss = 0.0
    num_batches = 0
    model.train()
    # for batch in train_data:
    for i, batch in enumerate(train_data):
        inputs = batch["embeddings"]
        ner_tags = batch["labels"]
        sentiments = batch["sentiments"]
        #print(inputs.shape, ner_tags.shape, sentiments.shape)
        inputs = inputs.to(torch.float32).to(device)
        ner_tags = ner_tags.long().to(device)  # NER targets (multi-class)
        sentiments = sentiments.long().to(
            device
        )  # Sentiment targets (binary)
        optimizer.zero_grad()
        ner_predictions, sentiment_predictions = model(inputs)
        # print(ner_predictions)
        # print(sentiment_predictions)

        """print(
            f"NER Predictions: {ner_predictions}, Sentiment Predictions: {sentiment_predictions}, Shapes: {ner_predictions.shape}, {sentiment_predictions.shape}"
        )"""

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
        ner_acc.update(ner_predictions, ner_tags)
        sentiment_acc.update(sentiment_predictions.squeeze(), sentiments)
        # print(f"NER Acc: {ner_acc.compute()}, Sentiment Acc: {sentiment_acc.compute()}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/ner_acc", ner_acc.compute(), epoch)
    writer.add_scalar("train/sentiment_acc", sentiment_acc.compute(), epoch)
    ner_acc.reset()
    sentiment_acc.reset()


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

    ner_acc = NERAccuracy()
    sentiment_acc = ClassificationAccuracy()
    total_loss = 0.0
    num_batches = 0
    model.eval()
    for i, batch in enumerate(val_data):
        inputs = batch["embeddings"]
        ner_tags = batch["labels"]
        sentiments = batch["sentiments"]
        inputs = inputs.to(torch.float32).to(device)
        ner_tags = ner_tags.long().to(device)
        sentiments = sentiments.long().to(device)
        ner_predictions, sentiment_predictions = model(inputs)

        """print(
            f"NER Predictions: {ner_predictions}, Sentiment Predictions: {sentiment_predictions}, Shapes: {ner_predictions.shape}, {sentiment_predictions.shape}"
        )"""
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
        ner_acc.update(ner_predictions, ner_tags)
        sentiment_acc.update(sentiment_predictions.squeeze(), sentiments)
        # print(f"NER Acc: {ner_acc.compute()}, Sentiment Acc: {sentiment_acc.compute()}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/ner_acc", ner_acc.compute(), epoch)
    writer.add_scalar("val/sentiment_acc", sentiment_acc.compute(), epoch)
    ner_acc.reset()
    sentiment_acc.reset()
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

    ner_acc = NERAccuracy()
    sentiment_acc = ClassificationAccuracy()
    model.eval()
    for i, batch in enumerate(test_data):
        inputs = batch["embeddings"]
        ner_tags = batch["labels"]
        sentiments = batch["sentiments"]
        inputs = inputs.to(torch.float32).to(device)
        ner_tags = ner_tags.long().to(device)
        sentiments = sentiments.long().to(device)
        ner_predictions, sentiment_predictions = model(inputs)
        # Update metrics
        ner_acc.update(ner_predictions, ner_tags)
        sentiment_acc.update(sentiment_predictions.squeeze(), sentiments)
    return {
        "ner_acc": ner_acc.compute(),
        "sentiment_acc": sentiment_acc.compute(),
    }


def combined_loss(
    ner_predictions,
    ner_targets,
    sa_predictions,
    sa_targets,
    ner_loss_fn,
    sa_loss_fn,
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
    
    ner_predictions = ner_predictions.view(-1, ner_predictions.shape[-1])
    ner_targets = ner_targets.view(-1)

    # sa_predictions = sa_predictions.view(-1, sa_predictions.shape[-1])
    #sa_targets = sa_targets.view(-1)


    loss_ner = ner_loss_fn(ner_predictions, ner_targets.long())
    loss_sa = sa_loss_fn(sa_predictions, sa_targets.long())
    # loss_sa = sa_loss_fn(sa_predictions.squeeze(), sa_targets.long())
    # print(f"NER Loss: {loss_ner.item()}, Sentiment Loss: {loss_sa.item()}")
    return loss_ner + loss_sa

