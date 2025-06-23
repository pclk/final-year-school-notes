from torch.utils.data import Dataset
import torch


class DrugReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length):
        self.reviews = reviews  # Changed from texts to reviews
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)  # Changed from texts to reviews

    def __getitem__(self, idx):
        text = str(self.reviews[idx])  # Using reviews instead of texts
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": label,
        }


class IndirectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Dataset class for indirect training data (e.g., IMDB)
        Args:
            texts: List of review texts
            labels: List of sentiment labels (0/1)
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class TextDatasetFromScratch(Dataset):
    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)  # Add this method

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Convert text to indices
        tokens = text.split()[: self.max_length]
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        # Pad sequence
        if len(indices) < self.max_length:
            indices.extend([self.vocab["<PAD>"]] * (self.max_length - len(indices)))

        return {
            "text": torch.tensor(indices, dtype=torch.long),
            "length": torch.tensor(len(tokens), dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
