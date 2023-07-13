import torch

from ..lstm_network import initialize_model
from ...helper_functions.elo_range import (
    rating_ranges, calculate_elo_range, guess_elo_from_range)
from ...helper_functions.load_dataset import load_games


# Switch to GPU if available for faster calculations
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")          # Use CUDA device
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")





def main():
    position_type = "boards_mirrors"
    dataset = load_games(position_type)
