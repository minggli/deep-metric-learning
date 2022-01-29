from torch.utils.data import DataLoader

from app.datasets import ExperimentDatasets, load_dataset

if __name__ == "__main__":
    train, test = load_dataset(ExperimentDatasets.MNIST)
