import torch


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
            correct += (logits.argmax(1) == y_test).type(torch.float32).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
