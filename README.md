# AutomaticAlertGen

## Proyecto Deep+NLP 
Rodrigo Covas, Mario Alonso, Marcos Garrido 

### Componentes del Modelo 
#### Reconocimiento de Entidades Nombradas (NER): 
- Datasets: CoNLL-2003 y OntoNotes 5.0. 
- Modelo: BERT (Embeddings) + Bi-LSTM. 

#### Análisis de Sentimientos (SA): 
- Datasets: Sentiment140 y Financial PhraseBank. 
- Modelo: BERT (Embeddings) + Bi-LSTM. 

#### Reconocimiento de Imágenes: 
- Dataset: ChatLabs AI Image Caption Generator. 
- Modelo: CNN + RNN.
  
### Roles y Tareas 
#### Adquisición de Datos: 
- Buscar y formatear datasets (texto, entidades nombradas, etiquetas de SA). 
- Data augmentation y preparación de datos para train/validation/test (guardados en formato Parquet). 
#### Diseñador de Modelo: 
- Prototipar modelos, elegir según resultados. 
- Diseñar funciones de pérdida, seleccionar hiperparámetros, entrenar modelos. 

#### Testeador: 
- Verificar robustez con tests unitarios y datos modificados. 

#### Documentador: 
- Preparar memoria en LaTeX y documentar fuentes/modelos. 

#### Revisor: 
- Revisar documentación, ortografía, y estándares de código (mypy, flake8). 

### Estándares de Código 
- Clases: PascalCase. 
- Métodos: snake_case. 
- Archivos: camelCase. 
- 1 archivo por clase; métodos con nombres declarativos y máx. 40 líneas. 
- Funciones reutilizables en utils.py. 

### Fases del Proyecto 
- Fase 0 (31 Marzo): Preparación y asignación de roles. 
- Fase 1 (5-10 Abril): Preparación de datos, pruebas con modelos preentrenados, elección de estructuras y embeddings. 
- Fase 2 (12 Abril): Entrenamiento inicial de modelos separados para NER y SA (objetivo: 7.0). 
- Fase 3 (19-21 Abril): Integración de modelos NER+SA, entrenamiento final (objetivo: 9.0). Si hay tiempo, incluir reconocimiento de imágenes. 
