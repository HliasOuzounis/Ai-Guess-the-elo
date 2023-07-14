import torch

from elo_guesser.helper_functions.get_device import get_device
device = get_device()


def train(model, x_train, y_train, optimizer, loss_func, num_epochs=10, validation_split=0.15):
    total_games = len(x_train)
    validation_games = int(total_games * validation_split)
    loss_graph = []

    x_val, y_val = x_train[-validation_games:], y_train[-validation_games:]
    x_train, y_train = x_train[:-validation_games], y_train[:-validation_games]

    for epoch in range(num_epochs):
        for _batch, (x, y) in enumerate(zip(x_train, y_train)):
            prediction = model.train(x)
            loss = loss_func(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation loss
        loss_sum = sum(loss_func(model(x, y).item())
                       for x, y, in zip(x_val, y_val))

        loss_graph.append(loss_sum/validation_games)
        print(f"Epoch: {epoch + 1}, Validation Loss: {loss_graph[-1]}")