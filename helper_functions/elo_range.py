import torch

start = 400
end = 3600
step = 200

rating_ranges = torch.stack([
    torch.Tensor((low, high)) for low, high in zip(range(start, end - step, step), range(start + step, end, step))
])


def calculate_elo_range(true_elo):
    stdev = 200
    norm_distribution = torch.distributions.Normal(true_elo, stdev)
    return norm_distribution.cdf(rating_ranges[:, 1]) - norm_distribution.cdf(rating_ranges[:, 0])


def guess_elo_from_range(probability_ranges):
    if probability_ranges.ndim == 1:
        probability_ranges = probability_ranges.unsqueeze(0)
        
    cum_probs = torch.cumsum(probability_ranges, dim=1)
    # Find the index where the cumulative probability is greater than 0.5, meaning we have passed the mean
    mean_index = torch.argmax((cum_probs > 0.5).int(), dim=1)
    # Assuming the probability density inside the range is constant
    # We can find the exact point where cum_prob = 0.5
    # We need a percentage of the range to have a cumulative probability of 0.5
    # We know the cum_prob until the previous range so we need 0.5 - cum_prob more
    # For that range, prob_needed = x * total_prob_in_range where x is a percentage of the range
    # rearranging:
    x = (0.5 - cum_probs[:, mean_index-1]) / probability_ranges[:, mean_index]
    # Adding the percentage of the range to the lower bound we get the value where cum_prob = 0.5
    return (rating_ranges[mean_index, 0] + x*step).diag().view(-1, 1)
