# Import necessary libraries
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple

# Set device to GPU if available, otherwise CPU
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT tokenizer and model for feature extraction
bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-cased")
bert_model: BertModel = BertModel.from_pretrained("bert-base-cased").to(DEVICE)
bert_model.eval()  # Set BERT model to evaluation mode

# Load a TorchScript model for NER and Sentiment Analysis (previously trained and saved)
model: torch.jit.ScriptModule = torch.jit.load("models/best_model.pt")
model.to(DEVICE)
model.eval()  # Set custom model to evaluation mode

# Load a pre-trained sequence-to-sequence model and tokenizer for text generation
# (e.g., FLAN-T5)
model_name: str = "google/flan-t5-base"
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
generator_model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(
    model_name
)

# Define label mappings for NER (Named Entity Recognition) and Sentiment Analysis
NER_LABELS: dict[int, str] = {
    0: "O",  # Outside any entity
    1: "Person",  # Person entity
    2: "Person",  # Person entity (possibly B/I tags merged)
    3: "Organization",  # Organization entity
    4: "Organization",  # Organization entity (possibly B/I tags merged)
    5: "Localization",  # Location entity
    6: "Localization",  # Location entity (possibly B/I tags merged)
    7: "O",  # Outside
    8: "O",  # Outside
}
SA_LABELS: dict[int, str] = {0: "positive", 1: "neutral", 2: "negative"}


def encode_sentence(sentence: str) -> torch.Tensor:
    """
    Tokenizes and encodes a sentence using BERT, returning the
    last hidden state (embeddings).
    Args:
        sentence (str): Input sentence.
    Returns:
        torch.Tensor: Embeddings of shape [1, seq_len, 768].
    """
    encoded = bert_tokenizer(
        sentence, return_tensors="pt", padding=True, truncation=True
    )
    input_ids: torch.Tensor = encoded["input_ids"].to(DEVICE)
    attention_mask: torch.Tensor = encoded["attention_mask"].to(DEVICE)
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state  # shape: [1, seq_len, 768]


def predict(sentence: str) -> Tuple[List[int], int]:
    """
    Predicts NER tags and sentiment class for a given sentence.
    Args:
        sentence (str): Input sentence.
    Returns:
        Tuple[List[int], int]: List of NER tag indices and sentiment class index.
    """
    embeddings: torch.Tensor = encode_sentence(sentence)
    with torch.no_grad():
        ner_logits, sa_logits = model(embeddings)

    ner_tags: List[int] = torch.argmax(ner_logits, dim=-1).squeeze(0).tolist()
    sa_class: int = int(torch.argmax(sa_logits, dim=-1).item())
    return ner_tags, sa_class


def decode_ner(sentence: str, ner_tags: List[int]) -> List[Tuple[str, str]]:
    """
    Converts NER tag indices to human-readable labels and pairs them with tokens.
    Args:
        sentence (str): Original sentence.
        ner_tags (List[int]): List of NER tag indices.
    Returns:
        List[Tuple[str, str]]: List of (token, label) pairs.
    """
    tokens: List[str] = bert_tokenizer.tokenize(sentence)
    return list(
        zip(tokens, [NER_LABELS.get(tag, "O") for tag in ner_tags[: len(tokens)]])
    )


def build_prompt(text: str, ner_decoded: List[Tuple[str, str]], sa_class: int) -> str:
    """
    Builds a prompt for the text generation model, including extracted entities
    and sentiment.
    Args:
        text (str): Original sentence.
        ner_decoded (List[Tuple[str, str]]): List of (token, label) pairs.
        sa_class (int): Sentiment class index.
    Returns:
        str: Formatted prompt string.
    """
    entities: List[str] = []
    for i in ner_decoded:
        if i[1] != "O":
            entities.append(i[0])
    entities_str: str = str(entities)
    sentiment_str: str = SA_LABELS[sa_class]

    prompt: str = (
        f"""Generate a sentence from the statement -{text}- with {sentiment_str}
        sentiment, including the words: {entities_str}"""
    )
    print(prompt)  # For debugging/monitoring
    return prompt


def main() -> None:
    """
    Main function to process sentences from a file, perform NER and sentiment analysis,
    build prompts, generate new sentences, and print results.
    """
    # Read sentences from file
    with open("data/sentences.txt", "r", encoding="utf-8") as f:
        lines: List[str] = [line.strip() for line in f if line.strip()]

    # Process each sentence
    for i, line in enumerate(lines):
        ner_tags, sa_class = predict(line)  # Get NER tags and sentiment
        ner_decoded = decode_ner(line, ner_tags)  # Convert tags to labels
        prompt = build_prompt(line, ner_decoded, sa_class)  # Build prompt for generator
        alert = generate_answer(prompt)  # Generate new sentence/alert
        print(f"\nSentence {i+1}: {line}")
        print(f"Sentiment: {SA_LABELS[sa_class]}")
        print("Entities:")
        print(ner_decoded)
        print(alert)


def generate_answer(prompt: str, max_tokens: int = 128) -> str:
    """
    Uses the text generation model to generate a new sentence based on the prompt.
    Args:
        prompt (str): Input prompt string.
        max_tokens (int): Maximum number of tokens to generate.
    Returns:
        str: Generated sentence.
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = generator_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    respuesta: str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return respuesta


# Entry point: Run main if this script is executed directly
if __name__ == "__main__":
    main()
