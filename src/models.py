import torch
import torch.nn as nn

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
    def __init__(self, input_dim, hidden_dim, num_layers, num_ner_classes, sentiment_classes=3, bidirectional=True):
        super(CombinedModel, self).__init__()
        self.ner_model = NERModel(input_dim, hidden_dim, num_layers, num_ner_classes, bidirectional)
        self.sa_model = SentimentModel(input_dim, hidden_dim, num_layers, sentiment_classes, bidirectional)

    def forward(self, embeddings):
        ner_logits = self.ner_model(embeddings)
        sentiment_logits = self.sa_model(embeddings)
        return ner_logits, sentiment_logits


"""class MyMultiTaskModel(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, bidirectional: bool, ner_tag_size: int) -> None:
        
        Constructor for the multi-task model.

        Args:
            hidden_size: Hidden size of the LSTM layers.
            num_layers: Number of LSTM layers.
            bidirectional: Whether the LSTM is bidirectional.
            ner_tag_size: Number of NER tags (output size for NER task).
        
        super(MyMultiTaskModel, self).__init__()
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers if not bidirectional else 2
        self.dropout: float = 0.2 if num_layers > 1 else 0.0
        
        # LSTM layer
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(
            input_size= 768, # BERT Embedding size
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=bidirectional,
        )
        
        lstm_output_size: int = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layer for NER tags
        self.fc_ner: torch.nn.Linear = torch.nn.Linear(lstm_output_size, ner_tag_size)
        
        # Fully connected layer for sentiment analysis (output size is 1)
        self.fc_sentiment: torch.nn.Linear = torch.nn.Linear(lstm_output_size, 1)

    def forward(self, inputs: torch.Tensor) -> tuple:
        
        Forward pass of the model.

        Args:
            inputs: Input tensor. Dimensions: [batch, length tokens, embeddings].

        Returns:
            A tuple containing:
                - NER tag predictions. Dimensions: [batch, ner_tag_size].
                - Sentiment analysis prediction. Dimensions: [batch, 1].
        
        batch_size: int = inputs.size(0)
        
        # Initialize hidden and cell states
        h0: torch.Tensor = torch.zeros(
            self.num_layers, batch_size, self.hidden_size
        ).to(inputs.device)
        c0: torch.Tensor = torch.zeros(
            self.num_layers, batch_size, self.hidden_size
        ).to(inputs.device)
        
        # Pass through LSTM
        out, _ = self.lstm(inputs, (h0, c0))
        
        # Use the output of the last time step for both tasks
        lstm_last_output = out[:, -1, :]  # Shape: [batch, lstm_output_size]
        
        # NER tag predictions
        ner_output = self.fc_ner(lstm_last_output)  # Shape: [batch, ner_tag_size]
        
        # Sentiment analysis prediction
        sentiment_output = self.fc_sentiment(lstm_last_output)  # Shape: [batch, 1]
        
        return ner_output, sentiment_output"""
