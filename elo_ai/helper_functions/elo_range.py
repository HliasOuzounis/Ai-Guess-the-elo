import torch

from elo_ai.helper_functions.get_device import get_device
device = get_device()

start = 0
end = 4000
step = 100


rating_ranges = torch.stack([
    torch.Tensor((low, high)) for low, high in zip(range(start, end - step, step), range(start + step, end, step))
]).int().to(device)


def get_rating_ranges():
    return rating_ranges


def calculate_rating_ranges(true_elo):
    if true_elo.ndim == 1:
        true_elo = true_elo.view(-1, 1)
    if true_elo.device != device:
        true_elo = true_elo.to(device)
    stdev = 200
    norm_distribution = torch.distributions.Normal(true_elo, stdev)
    return norm_distribution.cdf(rating_ranges[:, 1]) - norm_distribution.cdf(rating_ranges[:, 0])


def guess_elo_from_range(probability_ranges):
    # Find the average elo that would have played that game
    if probability_ranges.ndim == 1:
        probability_ranges = probability_ranges.unsqueeze(0)
    # Find the mid elo of each range and multiply it by the probability of that range to find the mean
    mid_rating_ranges = (rating_ranges[:, 0] + rating_ranges[:, 1]) / 2
    probability_ranges = probability_ranges.to(device)
    return torch.sum(probability_ranges * mid_rating_ranges, dim=1)
    


def round_elo(x):
    return 50 * torch.round(x / 50)


def convert_to_chessdotcom(prediction):
    # Taken from this discussion https://lichess.org/forum/general-chess-discussion/rating-conversion-formulae-lichessorg--chesscom
    # Considered only the blitz formula for simplicity
    return 1.138 * prediction - 665


def get_elo_prediction(predictions, is_chessdotcom=False, round=False):
    predictions = guess_elo_from_range(predictions)
    if is_chessdotcom:
        predictions = convert_to_chessdotcom(predictions)
    return round_elo(predictions).int().tolist() if round else predictions.int().tolist()


if __name__ == "__main__":
    test = torch.Tensor([600, 800, 850, 976, 2150]).to(device)
    predictions = guess_elo_from_range(test)
    print(guess_elo_from_range(predictions))
    print(round_elo(guess_elo_from_range(predictions)))
