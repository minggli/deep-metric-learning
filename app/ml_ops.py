import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from torch import nn

sns.set()

def is_module_crossentropy(module: nn.Module) -> bool:
    return type(getattr(module, "module", module)) == nn.CrossEntropyLoss


def _calculate_loss(output_1: torch.Tensor, output_2: torch.Tensor, y_train: torch.Tensor, loss_fn: nn.Module):
    if is_module_crossentropy(loss_fn):
        loss = loss_fn(output_2, y_train)
    else:
        loss = loss_fn(output_1, output_2, y_train)

    return loss


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_index, (x_train, y_train) in enumerate(dataloader):
        output_1, output_2 = model(x_train)
        loss = _calculate_loss(output_1, output_2, y_train, loss_fn)
        optimizer.zero_grad()
        # DataParallel
        joined_loss = loss.mean()
        joined_loss.backward()
        optimizer.step()

        if batch_index % 10 == 0:
            loss, current = joined_loss.item(), batch_index * len(x_train)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_single_batch(x_test, y_test, model, loss_fn):
    model.eval()
    with torch.no_grad():
        output_1, output_2 = model(x_test)
        loss = _calculate_loss(output_1, output_2, y_test, loss_fn)
        joined_loss = loss.mean()
        test_loss = joined_loss.item()

        if is_module_crossentropy(loss_fn):
            y_pred = output_2.argmax(dim=1).detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()
            f1 = f1_score(y_test, y_pred, average="macro")
            print(f"Test Error: \n f1 score: {f1:>4f}, loss: {test_loss:>8f} \n")
        else:
            print(f"Test Error: \n loss: {test_loss:>8f} \n")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x_test, y_test in dataloader:
            output_1, output_2 = model(x_test)
            loss = loss_fn(output_1, output_2, y_test)
            joined_loss = loss.mean()
            test_loss += joined_loss.item()
            correct += (output_1.argmax(1) == y_test).float32().sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def visualise_embedding(
    loss_name: str, dataset_name: str, epoch: int, images: list, x_test: torch.Tensor, y_test: torch.Tensor, model: nn.Module
):
    with torch.no_grad():
        model.eval()
        embedding_x, _ = model(x_test)
        array = embedding_x.cpu().numpy()
        label = np.asarray(list(map(str, y_test.cpu().numpy().astype("int").tolist())))

    sorting_index = np.argsort(label)
    label = label[sorting_index]
    array = array[sorting_index]

    reducer = TSNE(n_components=2, perplexity=30.0, n_iter=3000, learning_rate="auto", init='pca', random_state=0)
    reduced_array = reducer.fit_transform(array)

    fig = plt.figure(epoch, figsize=(12, 12), dpi=200)
    ax = fig.gca()
    sns.scatterplot(x=reduced_array[:, 0], y=reduced_array[:, 1], hue=label, ax=ax)
    ax.set_title(f"{dataset_name}, {loss_name}, Epoch: {epoch + 1}")
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])

    fig.canvas.draw()

    img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    images.append(img)
