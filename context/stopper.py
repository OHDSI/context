
class EarlyStopper:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.tolerance = 1e-5

    def update(self, loss: float) -> bool:
        if loss < self.best_loss - self.tolerance:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience