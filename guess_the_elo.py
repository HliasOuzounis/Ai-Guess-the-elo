import argparse
import os

import chess.engine
import chess.pgn
import torch

from models.lstm_network import LSTM
from analyze_game import (
    analyze_game, convert_position,
    fen_to_bitboard, fen_to_bitboard_mirror,
    fen_to_board, fen_to_board_mirror
)


position_converters = {
    "bitboards": fen_to_bitboard,
    "bitboards_mirrors": fen_to_bitboard_mirror,
    "boards": fen_to_board,
    "boards_mirrors": fen_to_board_mirror,
}
# position_type = "bitboards"
# position_type = "bitboards_mirrors"
# position_type = "boards"
position_type = "boards_mirrors"

mate_score = 1_000


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
        (200, 1000), (1000, 1200), (1200, 1400), (1400, 1600), (1600, 1800),
        (1800, 2000), (2000, 2200), (2200, 2400), (2400, 2600), (2600, 3600)
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

    if not os.path.isfile(engine_path):
        raise Exception("Could not find engine")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    position = convert_position(game, func=position_converters[position_type]).to(device)
    analysis = analyze_game(game, engine, mate_score).to(device)
    model_input = torch.cat((position, analysis), dim=-1)

    engine.close()

    input_size = position.shape[-1] + analysis.shape[-1]

    predict_single_output(is_chessdotcom, device, model_input, input_size)

    predict_rating_ranges(is_chessdotcom, device, model_input, input_size)


def predict_single_output(is_chessdotcom, device, model_input, input_size):
    max_elo = 3000
    single_output_model = LSTM(input_size, 128, 2, device, 1)
    single_output_model.load_state_dict(
        torch.load(f"models/single_output/{position_type}.pt")
    )
    prediction = single_output_model(model_input, train=False)[0] * max_elo

    if is_chessdotcom:
        prediction = convert_to_chessdotcom(prediction)
        
    print(f"Model1 prediction: {int(prediction[0].item())} for white and {int(prediction[1].item())} for black")


def predict_rating_ranges(is_chessdotcom, device, model_input, input_size):
    classes = 10
    rating_ranges_model = LSTM(input_size, 128, 2, device, classes)
    rating_ranges_model.load_state_dict(
        torch.load(f"models/rating_ranges/{position_type}.pt")
    )
    prediction = rating_ranges_model(model_input, train=False)
    prediction_w, prediction_b = tuple(guess_elo_range(pred) for pred in prediction[0])

    if is_chessdotcom:
        prediction_w = convert_to_chessdotcom(prediction_w)
        prediction_b = convert_to_chessdotcom(prediction_b)
        
    print(f"Model2 prediction: {prediction_w} for white and {prediction_b} for black")


if __name__ == "__main__":
    main()
