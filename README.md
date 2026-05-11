# AutomaticAlertGen
Rodrigo Covas, Mario Alonso, Marcos Garrido 

# Deep Learning NLP Project

This repository contains a modular deep learning pipeline for Named Entity Recognition (NER) and Sentiment Analysis (SA) using PyTorch and Hugging Face Transformers. The project is organized for easy experimentation, training, evaluation, and alert generation.

---

## Project Structure

- **src/data.py**  
  Handles data downloading, preprocessing, batching, and loading for both NER (CoNLL2003) and Sentiment Analysis (Financial Phrasebank). Implements custom PyTorch Dataset classes and DataLoader creation.

- **src/models.py**  
  Contains the `CombinedModel`, a multi-task LSTM-based neural network for both NER (token-level) and SA (sentence-level) with shared encoder and separate heads.

- **src/train_functions.py**  
  Implements training, validation, and testing steps, including loss/accuracy calculation, TensorBoard logging, and metric tracking for both tasks.

- **src/utils.py**  
  Utility functions and classes for:
  - Computing class weights for imbalanced data
  - Batch processing and saving/loading
  - Accuracy computation for NER and SA
  - Early stopping logic
  - Model saving/loading (TorchScript)
  - Setting random seeds for reproducibility

- **src/train.py**  
  Main script for sequential training:
  1. Trains the model on NER, saves the best checkpoint.
  2. Freezes the encoder and trains the model on Sentiment Analysis.
  3. Saves the final model.

- **src/evaluate.py**  
  Script for evaluating the trained model on the test sets for both NER and Sentiment Analysis, reporting accuracy.

- **src/alerts.py**  
  Script for generating alerts/sentences using the trained model.  
  **Usage:**  
  - Place a `sentences.txt` file with one sentence per line in the `./data` folder.
  - Run the script as described below.

---

## How to Use

1. **Install dependencies**  
   Make sure you have Python 3.8+ and install the required packages:
   pip install torch transformers datasets numpy tqdm tensorboard

2. **Preprocess and Train**
- To preprocess data and train the model, run:
  ```
  python -m src.train
  ```

3. **Evaluate**
- To evaluate the trained model on the test sets:
  ```
  python -m src.evaluate
  ```

4. **Generate Alerts**
- Place your input sentences in `./data/sentences.txt` (one sentence per line).
- To generate alerts or new sentences based on NER and Sentiment Analysis:
  ```
  python -m src.alert
  ```

---

## Code Quality

- The codebase is formatted and linted using:
- **Black Formatter** for consistent code style
- **MyPy** for static type checking
- **Flake8** for linting and code quality

---

## Notes

- All scripts should be run using the `python -m src.{file_name}` convention.
- Data and model checkpoints will be stored in the `./data` and `./models` directories, respectively.
- TensorBoard logs are saved in the `runs/` directory for easy experiment tracking.

---

## License

This project is provided for educational and research purposes.
