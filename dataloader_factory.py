from torch.utils.data import DataLoader
from transformers import default_data_collator

class DataLoaderFactory:
    def __init__(self, task_type: str = "nlp", batch_size=32, num_workers=0):
        if task_type not in ["nlp", "vision"]:
            raise ValueError("task_type must be 'nlp' or 'vision'")
        self.task_type = task_type
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _get_collate_fn(self):
        if self.task_type == "nlp":
            return default_data_collator
        else:
            return None

    def create_loaders(self, train_ds, val_ds, test_ds):
        collate_fn = self._get_collate_fn()

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

        return train_loader, val_loader, test_loader
