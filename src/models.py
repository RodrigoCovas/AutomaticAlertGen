import torch

class MyMultiTaskModel(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, bidirectional: bool, ner_tag_size: int) -> None:
        """
        Constructor for the multi-task model.

        Args:
            hidden_size: Hidden size of the LSTM layers.
            num_layers: Number of LSTM layers.
            bidirectional: Whether the LSTM is bidirectional.
            ner_tag_size: Number of NER tags (output size for NER task).
        """
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
        """
        Forward pass of the model.

        Args:
            inputs: Input tensor. Dimensions: [batch, length tokens, embeddings].

        Returns:
            A tuple containing:
                - NER tag predictions. Dimensions: [batch, ner_tag_size].
                - Sentiment analysis prediction. Dimensions: [batch, 1].
        """
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
        
        return ner_output, sentiment_output
