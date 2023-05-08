import chess.engine
import chess.pgn
import torch

from lstm_netwrok import LSTM
from analyze_games import analyze_game

import argparse


def get_game():
    parser = argparse.ArgumentParser(description="Guess the elo of a game")

    parser.add_argument(
        "pgn_file", type=str, metavar="pgn_file", help="The pgn file to analyze"
    )
    parser.add_argument(
        "--engine", type=str, metavar="engine-dir", help="Path of the engine to use", default="/usr/bin/stockfish"
    )
    parser.add_argument(
        "-c", action="store_true", help="True if the pgn file is from chess.com false otherwise"
    )
    args = parser.parse_args()
    
    return args.pgn_file, args.engine, args.c

# Based on probability distribution per rating range
# Try to guess the mean elo of the player
def guess_elo_range(elo_range):
    rating_ranges = [
        (400, 900),
        (900, 1100),
        (1100, 1300),
        (1300, 1500),
        (1500, 1700),
        (1700, 1900),
        (1900, 2100),
        (2100, 2300),
        (2300, 2500),
        (2500, 3000),
    ]
    s = 0
    for r, probability in zip(rating_ranges, elo_range):
        s += probability * (r[0] + r[1]) / 2

    return int(s / sum(elo_range))


def convert_to_chessdotcom(prediction):
    # Lichess ratings are usually around 400 points higher than chess.com
    return prediction - 400

def main():
    game_path, engine_path, is_chessdotcom = get_game()
    # game_path, engine_path, is_chessdotcom = "hlias_game.pgn", "/usr/bin/stockfish", False
    game = chess.pgn.read_game(open(game_path))
    if game is None:
        raise Exception("Not a valid pgn file")
    
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    analysis = analyze_game(game, engine, device)

    input_size = analysis.shape[-1]

    max_elo = 3000
    single_output_model = LSTM(input_size, 128, 2, device, 1)
    single_output_model.load_state_dict(torch.load("models/lstm_model_single_output.pt"))
    prediction = single_output_model(analysis, train=False)[0] * max_elo

    if is_chessdotcom:
        prediction = convert_to_chessdotcom(prediction)
    print(f"Model1 prediction: {int(prediction[0].item())} for white and {int(prediction[1].item())} for black")
    
    classes = 10
    rating_ranges_model = LSTM(input_size, 128, 2, device, classes)
    rating_ranges_model.load_state_dict(torch.load("models/lstm_model_rating_ranges.pt"))
    prediction_w, prediction_b = rating_ranges_model(analysis, train=False)[0]
    prediction_w = guess_elo_range(prediction_w)
    prediction_b = guess_elo_range(prediction_b)
    
    if is_chessdotcom:
        prediction_w = convert_to_chessdotcom(prediction_w)
        prediction_b = convert_to_chessdotcom(prediction_b)
    print(f"Model2 prediction: {prediction_w} for white and {prediction_b} for black")
    
    engine.close()
    
if __name__ == "__main__":
    main()