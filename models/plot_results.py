import torch
import matplotlib.pyplot as plt


def plot_loss(loss_graph):
    # Visualize the loss as the network trained
    # Should be a downward trend
    plt.plot(loss_graph)
    plt.title("Training loss plot: Rating Ranges model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # With 10 classes a random model would have -log(1/10) = 2.3 loss
    plt.savefig("temp/loss_plot.png")


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