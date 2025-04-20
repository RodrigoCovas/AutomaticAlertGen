import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
bert_model = BertModel.from_pretrained("bert-base-cased").to(DEVICE)
bert_model.eval()

model = torch.jit.load("models/best_model.pt")
model.to(DEVICE)
model.eval()

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

NER_LABELS = {0: "O", 1: "Person", 2: "Person", 3: "Organization", 4: "Organization", 5: "Localization", 6: "Localization", 7: "O", 8: "O"}
SA_LABELS = {0: "positive", 1: "neutral", 2: "negative"}

def encode_sentence(sentence):
    encoded = bert_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state  # shape: [1, seq_len, 768]

def predict(sentence):
    embeddings = encode_sentence(sentence)
    with torch.no_grad():
        ner_logits, sa_logits = model(embeddings)
    
    ner_tags = torch.argmax(ner_logits, dim=-1).squeeze(0).tolist()
    sa_class = torch.argmax(sa_logits, dim=-1).item()
    return ner_tags, sa_class

def decode_ner(sentence, ner_tags):
    tokens = bert_tokenizer.tokenize(sentence)
    return list(zip(tokens, [NER_LABELS.get(tag, "O") for tag in ner_tags[:len(tokens)]]))

def build_prompt(text, ner_decoded, sa_class):
    entities = []
    for i in ner_decoded:
        if i[1] != 'O':
            entities.append(i[0])
    entities_str = str(entities)
    sentiment_str = SA_LABELS[sa_class]

    prompt = f"Generate a sentence from the statement -{text}- with {sentiment_str} sentiment, including the words: {entities_str}"
    print(prompt)
    return prompt

def main():
    with open("data/frases.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, line in enumerate(lines):
        ner_tags, sa_class = predict(line)
        ner_decoded = decode_ner(line, ner_tags)
        prompt = build_prompt(line, ner_decoded, sa_class)
        alert = generate_answer(prompt)
        print(f"\nSentence {i+1}: {line}")
        print(f"Sentiment: {SA_LABELS[sa_class]}")
        print("Entities:")
        print(ner_decoded)
        print(alert)


def generate_answer(prompt: str, max_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = generator_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return respuesta

if __name__ == "__main__":
    main()



