import matplotlib.pyplot as plt
import numpy as np

from elo_ai.helper_functions.elo_range import get_elo_prediction, get_rating_ranges

rating_ranges = get_rating_ranges()[:, 0].tolist()


def plot_predictions(sequential_predictions, is_chessdotcom=False):
    current_index = 0
    max_index = len(sequential_predictions) - 1

    def on_key(event):
        nonlocal current_index
        if event.key == 'right' and current_index < max_index:
            current_index += 1
        elif event.key == 'left' and current_index > 0:
            current_index -= 1
        elif event.key in ('escape', 'q'):
            plt.close()
            return

        white_ax.cla()
        plot_index(axis=white_ax, color='blue', name="White")
        black_ax.cla()
        plot_index(axis=black_ax, color='orange', name="Black")

        plt.draw()

    def plot_index(axis, color, name):
        is_white =  0 if name == "White" else 1
        axis.bar(np.arange(len(rating_ranges)), sequential_predictions[current_index][is_white].tolist(), color=color)
        axis.set_title(f'{name}\nMove: {current_index + 1}, Prediction: {get_elo_prediction(sequential_predictions[current_index][is_white], is_chessdotcom)}')
        axis.set_xlabel('Rating Range')
        axis.set_ylabel('Probability')
        axis.set_xticks(range(0, len(rating_ranges), 5))
        axis.set_xticklabels(rating_ranges[::5])
        axis.set_ylim(0, 0.4)

    fig, (white_ax, black_ax) = plt.subplots(1, 2, figsize=(14, 7))
    fig.canvas.mpl_connect('key_press_event', on_key)

    plot_index(axis=white_ax, color='blue', name="White")
    plot_index(axis=black_ax, color='orange', name="Black")

    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    import torch
    sequential_predictions = torch.rand(10, 2, 16).to("cuda")
    plot_predictions(sequential_predictions)