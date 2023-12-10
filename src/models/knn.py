import torch

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return torch.tensor(predictions)

    def _predict(self, x):
        distances = torch.norm(self.X_train - x, dim=1)

        k_indices = torch.topk(distances, k=self.k, largest=False).indices

        k_nearest_labels = self.y_train[k_indices]

        prediction = torch.mode(k_nearest_labels).mode.item()
        return prediction