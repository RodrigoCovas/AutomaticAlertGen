from datasets import load_dataset

dataset = load_dataset("conll2003", trust_remote_code=True)
print(dataset)

example = dataset["train"][0]  # Access the first row of the training set
print(example)