from tqdm import tqdm
from tqdm import trange
import torch
from torch import optim
from data_processing import dataloader_test, dataloader_train


def train(model, optimizer, verbose=True):
    """
    This function trains a `model` on `train_loader` for 1 epoch and prints the
    loss value
    """
    device = torch.device("cpu")
    LOG_INTERVAL = 200
    model.train()
    train_loss = 0
    for batch_idx, (x, y) in enumerate(tqdm(dataloader_train, desc='Batches', leave=False)):
        x = x.flatten(start_dim=1).to(device).float()
        y = y.flatten(start_dim=0).to(device).float()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = model.get_loss(y, y_hat)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if verbose and batch_idx % LOG_INTERVAL == LOG_INTERVAL - 1:
            print('    Train [%d/%d]\t | \tLoss: %.5f' % (
                batch_idx * x.shape[0], len(dataloader_train.dataset), loss.item() / x.shape[0]))
    train_loss /= len(dataloader_train.dataset)
    if verbose:
        print('==> Train | Average loss: %.4f' % train_loss)


def test(model, verbose=True):
    """
    This function tests a `model` on a `test_loader` and prints the loss value
    """
    device = torch.device("cpu")
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for (x, y) in dataloader_test:
            x = x.flatten(start_dim=1).to(device).float()
            y_hat = model(x)
            loss = model.get_loss(y, y_hat)
            test_loss += loss.item()

    test_loss /= len(dataloader_test.dataset)
    if verbose:
        print('==> Test  | Average loss: %.4f' % test_loss)


def run(model, n_epoch, verbose=True):
    """
    This function will optimize parameters of `model` for `n_epoch` epochs
    on `train_loader` and validate it on `test_loader`.
    """
    LEARNING_RATE = 5e-5
    device = torch.device("cpu")
    # torch.cuda.set_device(device)
    # model.cuda()

    optimizer = None
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in trange(1, n_epoch + 1, desc='Epochs', leave=True):
        if verbose:
            print('\nEpoch %d:' % epoch)
        train(model, optimizer, verbose)
        test(model, verbose)
    return model
