import torch
from torch import nn
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_index, (x_train, y_train) in enumerate(dataloader):
        output_1, output_2 = model(x_train)
        loss = loss_fn(output_1, output_2, y_train)
        optimizer.zero_grad()
        # DataParallel
        joined_loss = loss.mean()
        joined_loss.backward()
        optimizer.step()

        if batch_index % 100 == 0:
            loss, current = joined_loss.item(), batch_index * len(x_train)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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


def visualise_embedding(epoch: int, images: list, x_test: torch.Tensor, y_test: torch.Tensor, model: nn.Module):
    with torch.no_grad():
        model.eval()
        embedding_x, output_2 = model(x_test)
        array = embedding_x.cpu().numpy()
        label = list(map(str, y_test.cpu().numpy().astype('int').tolist()))

    tsne = TSNE(n_components=2, perplexity=50., init='pca', random_state=0)
    reduced_array = tsne.fit_transform(array)

    fig = plt.figure(epoch, figsize=(12, 12), dpi=300)
    ax = fig.gca()
    sns.scatterplot(x=reduced_array[:, 0], y=reduced_array[:, 1], hue=label, ax=ax)
    ax.set_title(f"Epoch: {epoch + 1}")
    fig.canvas.draw()

    img = Image.frombytes('RGB', 
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb())

    images.append(img)

