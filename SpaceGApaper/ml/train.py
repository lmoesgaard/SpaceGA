import torch
import torch.nn as nn
import torch.optim as optim
import importlib

def get_model(model, arguments):
    module = importlib.import_module("ml")
    class_ = getattr(module, model)
    scorer = class_(arguments)
    return scorer

class train_model():
    def __init__(self, 
                 model_name,
                 model_path,
                 dataset, 
                 device,
                 patience=10,
                 n_epochs=100,
                 criterion=nn.MSELoss(), 
                 optimizer=optim.Adam
                ):
        model = get_model(model_name, 1024)
        self.model = model.to(device)
        self.model_path = model_path
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters())
        self.n_epochs = n_epochs
        self.patience = patience
        self.device = device
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.mode = "train"

    def train_model(self):
        for epoch in range(self.n_epochs):
            self.mode = "train"
            self.model.train()
            train_loss = self.run_batches()

            with torch.no_grad():
                self.mode = "val"
                self.model.eval()
                val_loss = self.run_batches()

            #print(f'Epoch {epoch+1}/{self.n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if self.early_stopping(val_loss):
                break

        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to("cpu")


    def run_batches(self):
        self.dataset.change_mode(self.mode)
        acc_loss = 0.0
        nmol = 0
        inputs, targets = self.dataset.get_batch()
        while inputs is not None:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs.flatten(), targets.flatten())
            if self.mode == "train":
                loss.backward()
                self.optimizer.step()
            acc_loss += loss.item() * inputs.size(0)
            nmol += inputs.size(0)
            inputs, targets = self.dataset.get_batch()
        acc_loss /= nmol
        return acc_loss

    def early_stopping(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            torch.save(self.model.state_dict(), self.model_path)
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            #print('Early stopping triggered.')
            return True
        else:
            return False
