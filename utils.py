import torch
import gc
from tqdm import tqdm
import optuna
from optuna.trial import Trial
from torch.utils.data import DataLoader
import torch.optim as optim
from adopt import ADOPT
from dataloader_factory import DataLoaderFactory

from tabulate import tabulate

def get_optimizer(optimizer_name, model, optimizer_params=None):
    if optimizer_name == "SGD":
        return optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_name == "Adam":
        return optim.Adam(
            model.parameters(), **optimizer_params,
        )
    elif optimizer_name == "AdamW":
        return optim.AdamW(
            model.parameters(),**optimizer_params)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(),**optimizer_params)
    elif optimizer_name == "ADOPT":
        return ADOPT(model.parameters(),**optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def objective(trial: optuna.Trial,
              optimizer_name: str = "Adam",
              train_dataset=None,
              val_dataset=None,
              num_classes=2,
              task_type: str = "classification",
              model_fn=None,
              device: str = "cuda"):

    if optimizer_name not in ["SGD", "Adam", "AdamW", "RMSprop", "ADOPT"]:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    if train_dataset is None or val_dataset is None:
        raise ValueError("train_dataset and val_dataset must be provided")
    if model_fn is None:
        raise ValueError("You must provide model_fn")

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 1, 10)
    optimizer_params = {"lr": learning_rate}

    if hasattr(train_dataset, 'set_format'):
        train_dataset.set_format(None)
        val_dataset.set_format(None)

    class HuggingFaceDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        def __getitem__(self, idx):
            item = self.dataset[idx]
            return {k: torch.tensor(v) if isinstance(v, (list, int)) and not isinstance(v, str) else v
                    for k, v in item.items()}
        def __len__(self):
            return len(self.dataset)

    if task_type == "nlp":
        wrapped_train = HuggingFaceDatasetWrapper(train_dataset)
        wrapped_val = HuggingFaceDatasetWrapper(val_dataset)
    elif task_type == "vision":
        wrapped_train = train_dataset
        wrapped_val = val_dataset

    loader_factory = DataLoaderFactory(task_type=task_type, batch_size=batch_size)
    train_loader, val_loader, _ = loader_factory.create_loaders(
        train_ds=wrapped_train,
        val_ds=wrapped_val,
        test_ds=None
    )

    model = model_fn(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model, optimizer_params=optimizer_params)

    try:
        for epoch in range(num_epochs):
            model.train()
            progress_bar = tqdm(train_loader, desc=f"Trial {trial.number} | Epoka {epoch+1}/{num_epochs}", leave=False)

            for batch in progress_bar:
                try:
                    if task_type == "vision":
                        inputs, labels = batch
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)

                    elif task_type == "nlp":
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['label'].to(device)
                        outputs = model(input_ids, attention_mask=attention_mask).logits

                    else:
                        raise ValueError(f"Unsupported task type: {task_type}")

                    optimizer.zero_grad()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    progress_bar.set_postfix(loss=loss.item())

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        gc.collect()
                        raise optuna.exceptions.TrialPruned()
                    else:
                        raise

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if task_type == "vision":
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                elif task_type == "nlp":
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask).logits

                else:
                    raise ValueError(f"Unsupported task type: {task_type}")

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.exceptions.TrialPruned()
        else:
            raise

    torch.cuda.empty_cache()
    gc.collect()

    return accuracy

def tune_optimizers_with_optuna(optimizers,
                                n_trials=5,
                                train_dataset=None,
                                val_dataset=None,
                                num_classes=2,
                                task_type: str = "classification",
                                model_fn=None):

    if train_dataset is None or val_dataset is None:
        raise ValueError("train_dataset and val_dataset must be provided")

    best_hyperparams = {}

    for optimizer_name in optimizers:
        print(f"Hyperparameter tuning for: {optimizer_name}")

        def wrapped_objective(trial):
            torch.cuda.empty_cache()
            gc.collect()
            return objective(trial, 
                             optimizer_name, 
                             train_dataset, 
                             val_dataset, 
                             num_classes, 
                             task_type, 
                             model_fn)

        study = optuna.create_study(direction="maximize")
        study.optimize(wrapped_objective, n_trials=n_trials)

        best_hyperparams[optimizer_name] = {
            "params": study.best_params,
            "accuracy": study.best_value
        }

        print(f"Best hyperparameters for {optimizer_name}: {study.best_params}")

    return best_hyperparams

def print_hyperparams_table(hyperparams):
    headers = ["Optimizer", "Best Accuracy", "Best Hyperparameters"]
    table = []

    for optimizer, params in hyperparams.items():
        table.append([
            optimizer,
            f"{params['accuracy']:.4f}",
            str(params['params'])
        ])

    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))