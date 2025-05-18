from tqdm import tqdm
from utils import get_optimizer
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torch.nn as nn

def evaluate_model(model, test_loader, task_type="nlp", device="cuda"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if task_type == "nlp":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits

            elif task_type == "vision":
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1

def train_model_with_neptune(optimizer_name, model, criterion, train_loader, run, epochs, task_type="nlp", device = "cuda", optimizer_params=None):
    model = model.to(device)
    optimizer = get_optimizer(optimizer_name, model, optimizer_params)
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} ({optimizer_name})", leave=False)

        for batch in progress_bar:
            optimizer.zero_grad()

            if task_type == "nlp":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits

            elif task_type == "vision":
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss_so_far = epoch_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=f"{avg_loss_so_far:.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)
        run[f"{optimizer_name}/loss"].log(avg_epoch_loss)

    return losses

def log_evaluation_metrics_to_neptune(run, optimizer_name, accuracy, precision, recall, f1):
    run[f"{optimizer_name}/accuracy"].log(accuracy)
    run[f"{optimizer_name}/precision"].log(precision)
    run[f"{optimizer_name}/recall"].log(recall)
    run[f"{optimizer_name}/f1_score"].log(f1)

def compare_optimizers_with_metrics(optimizers, run, model_fn, train_loader, test_loader, 
                                    task_type="nlp", num_classes=2, epochs=10, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    results = {}

    for optimizer_name, optimizer_params in optimizers.items():
        model = model_fn(num_classes=num_classes).to(device)
        losses = train_model_with_neptune(optimizer_name, model, criterion, 
                                          train_loader, run, epochs, 
                                          task_type, device=device,
                                          optimizer_params=optimizer_params)
        
        accuracy, precision, recall, f1 = evaluate_model(model, test_loader, task_type, device=device)
        results[optimizer_name] = {
            "losses": losses,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        log_evaluation_metrics_to_neptune(run, optimizer_name, accuracy, precision, recall, f1)

    plt.figure(figsize=(10, 6))
    for optimizer_name, metrics in results.items():
        plt.plot(range(epochs), metrics["losses"], label=f"{optimizer_name} Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Comparison of Optimizers on {task_type.upper()}')
    plt.legend()
    plt.show()

    for optimizer_name, metrics in results.items():
        print(f"Optimizer: {optimizer_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("-" * 30)
