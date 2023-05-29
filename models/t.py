import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the parameters for the normal distribution
mean = 1700
std = 200

# Generate x values for the normal distribution
x = np.linspace(mean - 3 * std, mean + 3 * std, 100)

# Compute the probability density function (PDF) of the normal distribution
pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2)) * 200

# Define the rating ranges and their corresponding probabilities
# Define the rating ranges
rating_ranges = [
    (400, 1000), (1000, 1200), (1200, 1400), (1400, 1600), (1600, 1800),
    (1800, 2000), (2000, 2200), (2200, 2400), (2400, 2600), (2600, 3000)
]
# Compute the cumulative probabilities for each rating range
cumulative_probs = []
for start, end in rating_ranges:
    prob = norm.cdf(end, mean, std) - norm.cdf(start, mean, std)
    cumulative_probs.append(prob)

print(cumulative_probs)

plt.grid()
# Plot the normal distribution
plt.plot(x, pdf, label='Normal Distribution')

# Plot the bar plot with probabilities for rating ranges
for i, (start, end) in enumerate(rating_ranges):
    plt.bar((start + end) / 2, cumulative_probs[i], width=end - start, align='center', alpha=0.5)

# Set labels and title
plt.xlabel('Rating')
plt.ylabel('Probability')
plt.title('Normal Distribution with Rating Ranges')

# Add legend
plt.legend()

# Show the plot
plt.savefig('../models/loss_plots/normal_distribution.png')
plt.show()