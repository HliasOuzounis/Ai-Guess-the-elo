import matplotlib.pyplot as plt
import torch

from elo_guesser.helper_functions import elo_range

fig, axs = plt.subplots(2, 1, sharex=True)
stdev = 200

elos = [1300, 1730, 2650]
cols = ["r", "g", "b"]
fig.suptitle("Red: 1300 elo\nGreen: 1730 elo\nBlue: 2650 elo")

for elo, color in zip(elos, cols):
    probabilities = elo_range.calculate_rating_ranges(torch.Tensor([elo]))
    norm_distribution = torch.distributions.Normal(elo, stdev)
    rating_ranges = elo_range.get_rating_ranges().cpu().numpy()[:, 0]

    
    axs[0].set_title("Probability of Rating Ranges")
    axs[0].bar(rating_ranges, probabilities.view(-1).cpu().numpy(), color=color, width=200)
    axs[1].set_title("Probability Distribution")
    axs[1].plot(torch.arange(200, 3600), norm_distribution.log_prob(torch.arange(200, 3600)).exp().cpu().numpy(), color=color)

plt.show()