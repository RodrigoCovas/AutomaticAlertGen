import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class NERModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=True):
        super(NERModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, embeddings):
        """
        Args:
            embeddings: shape (batch_size, seq_len, input_dim)
        Returns:
            shape (batch_size, seq_len, num_classes)
        """
        # LSTM outputs all hidden states (one per token)
        lstm_out, _ = self.lstm(embeddings)  # shape: (batch_size, seq_len, hidden_dim*num_directions)

        # Classify each token
        logits = self.classifier(lstm_out)  # shape: (batch_size, seq_len, num_classes)
        return logits
    

class SentimentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes=3, bidirectional=True):
        super(SentimentModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, embeddings):
        """
        Args:
            embeddings: shape (batch_size, seq_len, input_dim)
        Returns:
            shape (batch_size, num_classes)
        """
        
        _, (h_n, _) = self.lstm(embeddings)  # h_n: (num_layers * num_directions, batch_size, hidden_dim)

        # Concatenate last hidden states from all directions
        if self.bidirectional:
            h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch_size, hidden_dim*2)
        else:
            h_n = h_n[-1]  # (batch_size, hidden_dim)

        output = self.classifier(h_n)  # (batch_size, num_classes)
        return output


class CombinedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_ner_classes=10, sentiment_classes=3, bidirectional=True):
        super(CombinedModel, self).__init__()
        self.ner_model = NERModel(input_dim, hidden_dim, num_layers, num_ner_classes, bidirectional)
        self.sa_model = SentimentModel(input_dim, hidden_dim, num_layers, sentiment_classes, bidirectional)

    def forward(self, embeddings):
        ner_logits = self.ner_model(embeddings)
        sentiment_logits = self.sa_model(embeddings)
        return ner_logits, sentiment_logits







