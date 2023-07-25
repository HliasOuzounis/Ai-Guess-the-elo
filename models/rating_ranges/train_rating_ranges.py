import torch
import matplotlib.pyplot as plt

from elo_guesser.helper_functions import load_dataset
from elo_guesser.helper_functions import elo_range
from elo_guesser.models import lstm_network

# Switch to GPU if available for faster calculations
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")  # Use CUDA device
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")

rating_ranges = elo_range.get_rating_ranges().to(device)


def modify_dataset(dataset):
    import random

    random.shuffle(dataset)

    total_games = len(dataset)
    test_games = int(total_games * 0.15)

    x_train = [game.to(device) for game, _ in dataset[:-test_games]]
    y_train = [elo_range.guess_elo_from_range(elo).to(device)
               for _, elo in dataset[:-test_games]]
    x_test = [game.to(device) for game, _ in dataset[-test_games:]]
    y_test = [elo_range.guess_elo_from_range(elo).to(device)
              for _, elo in dataset[-test_games:]]

    return x_train, y_train, x_test, y_test


def create_model(input_size):
    hidden_size = 128
    num_layers = 2
    learning_rate = 0.001
    lstm_model, optimizer = lstm_network.initialize_model(
        input_size, hidden_size, num_layers, device, learning_rate, num_classes=len(rating_ranges))
    return lstm_model, optimizer


def train_model(model, optimizer, x_train, y_train, epochs=10):
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_graph = lstm_network.train_model(
        model, optimizer, loss_fn, (x_train, y_train), epochs)
    return loss_graph


def plot_results(lstm_model, x_test, y_test, loss_graph):
    plot_loss(loss_graph)
    predictions = get_predictions(lstm_model, x_test)
    compare_results(predictions, y_test)
    plot_predictions(predictions, y_test)


def plot_loss(loss_graph):
    # Visualize the loss as the network trained
    # Should be a downward trend
    plt.plot(loss_graph)
    plt.title("Training loss plot: Rating Ranges model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # With 10 classes a random model would have -log(1/10) = 2.3 loss
    plt.savefig("temp/loss_plot.png")


def get_predictions(lstm_model, x_test):
    predictions = []
    for game in x_test:
        prediction, (_h, _c) = lstm_model(game)
        prediction = elo_range.guess_elo_from_range(prediction)
        predictions.append(prediction)
    return predictions


def get_cutoff_value(tensor, percentage):
    cutoff = int(percentage * len(tensor))
    cutoff_value, _ = torch.kthvalue(tensor, cutoff)
    return cutoff_value.item()


def compare_results(predictions, y_test):
    predictions = torch.cat(predictions).view(-1)
    y_test = torch.cat(y_test).view(-1)

    differences = torch.abs(predictions - y_test)
    fiftieth_percentile = get_cutoff_value(differences, 0.5)
    print(f"50th percentile: {fiftieth_percentile}")
    ninetieth_percentile = get_cutoff_value(differences, 0.9)
    print(f"90th percentile: {ninetieth_percentile}")

    plot_predictions(predictions, y_test, fiftieth_percentile,
                     ninetieth_percentile)


def plot_predictions(predictions, y_test, fifty=None, ninety=None):
    fig, axs = plt.subplots(1, 1)
    axs.plot(y_test[:2000], predictions[:2000], 'o', label="Predicted vs Real")

    # the closer to this line the better
    axs.plot([800, 2800], [800, 2800], 'r-', label="Perfect prediction")
    axs.plot([800, 2800], [800 + fifty, 2800 + fifty],
             'g--', label="200 points error")
    # these two lines show acceptable error (200 elo)
    axs.plot([800, 2800], [800 - fifty, 2800 - fifty], 'g--', label="")

    axs.plot([800, 2800], [800 + ninety, 2800 + ninety],
             'm--', label="90% of data")
    axs.plot([800, 2800], [800 - ninety, 2800 - ninety], 'm--',
             label="")  # these two lines encompass 90% of the data

    axs.set_title(f"Real vs Predicted\nRating Ranges model")

    axs.set_xlabel("Real")
    axs.set_ylabel("Predicted")

    axs.set_xlim(600, 3000)
    axs.set_ylim(600, 3000)

    axs.grid()
    axs.legend()

    fig.savefig("temp/predictions.png")
    plt.show()


def main():
    position_type = "boards_mirrors"
    dataset = load_dataset.load_games(position_type)
    print(f"Loaded {len(dataset)} games")
    x_train, y_train, x_test, y_test = modify_dataset(dataset)
    input_size = x_train[0].shape[2]

    lstm_model, optimizer = create_model(input_size)
    print(f"Training model with {len(x_train)} games")
    loss_graph = train_model(lstm_model, optimizer,
                             x_train, y_train, epochs=10)
    plot_results(lstm_model, x_test, y_test, loss_graph)


if __name__ == "__main__":
    main()
