import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class EmotionDataset(Dataset):
    """Custom Dataset for loading emotion images with one-hot encoded classes from CSV"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Read the classes CSV file
        csv_path = os.path.join(root_dir, "_classes.csv")
        self.df = pd.read_csv(csv_path)
        # Clean column names by removing leading/trailing spaces
        self.df.columns = self.df.columns.str.strip()

        # Define emotion classes (columns except 'filename')
        self.classes = [col for col in self.df.columns if col != "filename"]

        # Get all image files and their corresponding labels
        self.images = []
        self.labels = []

        for _, row in self.df.iterrows():
            img_path = os.path.join(root_dir, row["filename"])
            if os.path.exists(img_path):
                self.images.append(img_path)
                # Get the index where 1 appears in the one-hot encoded row
                label = [row[cls] for cls in self.classes].index(1)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
