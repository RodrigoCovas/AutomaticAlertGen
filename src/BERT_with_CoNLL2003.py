from transformers import BertTokenizerFast, BertModel
import torch

# Cargar el tokenizador rápido y el modelo BERT
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

# Ejemplo de tokens pre-segmentados
tokens = ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']

# Tokenizar respetando la segmentación original, incluyendo el offset_mapping
inputs = tokenizer(tokens, is_split_into_words=True, return_tensors='pt', return_offsets_mapping=True)

# Extraer y eliminar offset_mapping del diccionario para que no se pase al modelo
offset_mapping = inputs.pop("offset_mapping")

# Pasar el diccionario modificado al modelo
outputs = model(**inputs)
token_embeddings = outputs.last_hidden_state

# Convertir los IDs de tokens a tokens legibles
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print("Tokens:", tokens)

print("Shape de token_embeddings:", token_embeddings.shape)
