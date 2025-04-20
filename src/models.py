import torch
import torch.nn as nn
from typing import Tuple


class CombinedModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_ner_classes: int,
        sentiment_classes: int = 3,
        bidirectional: bool = True,
    ) -> None:
        """
        A multi-task model for both Named Entity Recognition (NER) and
        Sentiment Analysis (SA).
        Uses a shared LSTM encoder, with separate heads for NER (token-level)
        and SA (sentence-level).

        Args:
            input_dim (int): Dimension of input embeddings.
            hidden_dim (int): Hidden size for the LSTM.
            num_layers (int): Number of LSTM layers.
            num_ner_classes (int): Number of NER tag classes.
            sentiment_classes (int): Number of sentiment classes (default: 3).
            bidirectional (bool): If True, use a bidirectional LSTM.
        """
        super(CombinedModel, self).__init__()
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.bidirectional: bool = bidirectional
        self.num_directions: int = 2 if bidirectional else 1

        # Shared LSTM encoder for sequence modeling
        self.encoder: nn.LSTM = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # NER head: token-level classifier (per token in sequence)
        self.ner_classifier: nn.Linear = nn.Linear(
            hidden_dim * self.num_directions, num_ner_classes
        )
        # SA head: sentence-level classifier (on last hidden state)
        self.sa_classifier: nn.Linear = nn.Linear(
            hidden_dim * self.num_directions, sentiment_classes
        )

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the model.

        Args:
            embeddings (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim)
                or (batch, input_dim) (single vector per batch).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ner_logits: (batch, seq_len, num_ner_classes)
                - sa_logits: (batch, sentiment_classes)
        """
        if embeddings.dim() == 2:
            # If input is (batch, input_dim), add a sequence length dimension
            embeddings = embeddings.unsqueeze(1)
        # Ensure LSTM weights are contiguous in memory for efficiency
        self.encoder.flatten_parameters()
        # Pass through LSTM: lstm_out is (batch, seq_len, hidden_dim*num_directions)
        lstm_out, (h_n, _) = self.encoder(embeddings)

        # NER: classify each token's representation
        ner_logits: torch.Tensor = self.ner_classifier(
            lstm_out
        )  # (batch, seq_len, num_ner_classes)

        h_n_cat: torch.Tensor
        # SA: use the last hidden state(s) from the LSTM for sentence classification
        if self.bidirectional:
            # For bidirectional LSTM, concatenate the last forward and
            # backward hidden states
            forward: torch.Tensor = h_n[-2, :, :]  # (batch, hidden_dim)
            backward: torch.Tensor = h_n[-1, :, :]  # (batch, hidden_dim)
            h_n_cat = torch.cat((forward, backward), dim=1)  # (batch, hidden_dim*2)
        else:
            # For unidirectional LSTM, use the last layer's hidden state
            h_n_cat = h_n[-1]  # (batch, hidden_dim)

        # SA: classify the sentence representation
        sa_logits: torch.Tensor = self.sa_classifier(
            h_n_cat
        )  # (batch, sentiment_classes)
        return ner_logits, sa_logits
