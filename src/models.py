import torch
import torch.nn as nn


class CombinedModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        num_ner_classes,
        sentiment_classes=3,
        bidirectional=True,
    ):
        super(CombinedModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Shared LSTM encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # NER head: token-level classifier
        self.ner_classifier = nn.Linear(
            hidden_dim * self.num_directions, num_ner_classes
        )
        # SA head: sentence-level classifier
        self.sa_classifier = nn.Linear(
            hidden_dim * self.num_directions, sentiment_classes
        )

    def forward(self, embeddings):
        if embeddings.dim() == 2:
            # If input is (batch, input_dim), add seq_len dimension
            embeddings = embeddings.unsqueeze(1)
        # embeddings: (batch, seq_len, input_dim)
        lstm_out, (h_n, _) = self.encoder(
            embeddings
        )  # lstm_out: (batch, seq_len, hidden_dim*num_directions)

        # NER: classify each token
        ner_logits = self.ner_classifier(lstm_out)  # (batch, seq_len, num_ner_classes)

        # SA: use the last hidden state(s)
        if self.bidirectional:
            # Get last layer's forward and backward hidden states
            forward = h_n[-2, :, :]  # (batch, hidden_dim)
            backward = h_n[-1, :, :]  # (batch, hidden_dim)
            h_n_cat = torch.cat((forward, backward), dim=1)  # (batch, hidden_dim*2)
        else:
            h_n_cat = h_n[-1, :, :]  # (batch, hidden_dim)
        sentiment_logits = self.sa_classifier(h_n_cat)  # (batch, sentiment_classes)

        return ner_logits, sentiment_logits
