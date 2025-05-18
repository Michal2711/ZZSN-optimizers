from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from torchvision.datasets import CIFAR100
from torchvision import transforms
import torch
import pandas as pd
import numpy as np
import random

class DatasetPipeline:
    def __init__(self, seed=42):
        self.seed = seed
        self.tokenizer = None
        self.task_type = None

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def prepare_nlp(self, csv_path, tokenizer_name="distilbert-base-uncased",
                    train_size=2000, test_size=250, val_size=250):
        self.task_type = "nlp"
        self.set_seed()

        df = pd.read_csv(csv_path, encoding='latin-1')
        df['sentiment'] = df['sentiment'].astype('category')

        df_train, df_temp = train_test_split(
            df, train_size=train_size, test_size=test_size + val_size,
            stratify=df['sentiment'], random_state=self.seed
        )
        df_test, df_val = train_test_split(
            df_temp, train_size=test_size, test_size=val_size,
            stratify=df_temp['sentiment'], random_state=self.seed
        )

        df_train.reset_index(drop=True, inplace=True)
        df_val.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
        
        label_map = {'negative': 0, 'positive': 1}
        for df_split in [df_train, df_val, df_test]:
            df_split.rename(columns={"sentiment": "label"}, inplace=True)
            df_split["label"] = df_split["label"].map(label_map)

        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

        def preprocess_function(examples):
            return self.tokenizer(
                examples["review"],
                truncation=True,
                padding="max_length",
                max_length=512
            )

        train_ds = Dataset.from_pandas(df_train[["review", "label"]])
        val_ds = Dataset.from_pandas(df_val[["review", "label"]])
        test_ds = Dataset.from_pandas(df_test[["review", "label"]])

        train_ds = train_ds.map(preprocess_function, batched=True)
        val_ds = val_ds.map(preprocess_function, batched=True)
        test_ds = test_ds.map(preprocess_function, batched=True)

        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        return train_ds, val_ds, test_ds

    def prepare_vision(self, dataset_class=CIFAR100, root="/data"):
        self.task_type = "vision"
        self.set_seed()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        full_train = dataset_class(root=root, train=True, transform=transform, download=True)
        test_dataset = dataset_class(root=root, train=False, transform=transform, download=True)

        train_size = int(0.8 * len(full_train))
        val_size = len(full_train) - train_size
        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train, [train_size, val_size], generator=generator
        )

        return train_dataset, val_dataset, test_dataset
