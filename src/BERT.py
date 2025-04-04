from transformers import BertTokenizer, BertModel
import torch

# Cargar el tokenizador y el modelo BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

# Texto de ejemplo
text = "Um dia um macaco se casou com a borboleta, nasceu um barbolecaco e uma macacoleta, que confução, que confução, cabeça de cachorro com asas de dragão"
inputs = tokenizer(text, return_tensors='pt')

# Obtener las salidas del modelo
outputs = model(**inputs)
token_embeddings = outputs.last_hidden_state

# Mostrar la forma del tensor de embeddings: debe ser [batch_size, sequence_length, hidden_size]
print("Shape de token_embeddings:", token_embeddings.shape)

# Convertir los IDs de tokens a tokens legibles
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print("Tokens:", tokens)

# Para SA: imprimir el embedding del token [CLS] (posición 0)
cls_embedding = token_embeddings[:, 0, :]
print("Embedding del token [CLS]:", cls_embedding)

# Para NER: imprimir la forma del tensor (excluyendo [CLS] y [SEP] si se desea)
ner_embeddings = token_embeddings[:, 1:-1, :]
print("Shape de ner_embeddings:", ner_embeddings.shape)
 