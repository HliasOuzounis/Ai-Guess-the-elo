import argparse
import os
import torch

import chess.engine
import chess.pgn

from elo_ai.models import complex_network
from elo_ai.helper_functions import game_analysis, position_converters
from elo_ai.helper_functions.elo_range import guess_elo_from_range, round_elo

from elo_ai.helper_functions.get_device import get_device
device = get_device()

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
    # parser.add_argument(
    #     "--username", type=str, metavar="username", help="The username of the player ou want predictions for", default=None
    # )
    args = parser.parse_args()

    return args.pgn_file, args.engine, args.c


def convert_to_chessdotcom(prediction):
    # Taken from this forum https://lichess.org/forum/general-chess-discussion/rating-conversion-formulae-lichessorg--chesscom
    # Considered only the blitz formula for simplicity
    return 1.138 * prediction - 665


def load_model():
    input_size = 17
    channels = 2
    classes = 16
    model = complex_network.EloGuesser(
        input_size, input_channels=channels, num_classes=classes)
    model.load_state_dict(torch.load(
        "elo_ai/models/rating_ranges/boards_mirrors.pt"))
    return model


def main():
    game_path, engine_path, is_chessdotcom = get_game()

    if not os.path.isfile(engine_path):
        raise Exception("Could not find engine")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    games = []
    game_index = 0
    with open(game_path) as pgn:
        while True:
            game_index += 1
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games.append(game)
            predict_game(is_chessdotcom, game, engine, game_index)

    engine.close()


def predict_game(is_chessdotcom, game, engine, game_index):

    analysis = game_analysis.analyze_game(
        game, engine, progress_bar=True, time_limit=0.1)

    func = position_converters.fen_to_board_mirror
    positions, _elo = position_converters.convert_position(game, func)

    predictions = get_ai_prediction(
        is_chessdotcom, (positions.to(device), analysis.to(device)))

    final_predictions = predictions[-1]
    print(
        f"Models predictions for game {game_index} are: \n{final_predictions[0][0]} for white\n{final_predictions[1][0]} for black")

    return final_predictions


def get_ai_prediction(is_chessdotcom, game):
    model = load_model()
    positions, analysis = game
    c, h = None, None
    moves = analysis.size(1)

    predictions = []
    for move in range(moves):
        pos, evaluation = (positions[:, move].unsqueeze(1),
                           analysis[:, move].unsqueeze(1))
        prediction, (h, c) = model.eval()((pos, evaluation), h, c)
        elo_predictions = guess_elo_from_range(prediction)
        if is_chessdotcom:
            elo_predictions = convert_to_chessdotcom(elo_predictions)
        predictions.append(round_elo(elo_predictions).int().tolist())

    return predictions


if __name__ == "__main__":
    main()
    # model = load_model()
    # torch.onnx.export(model, (torch.zeros(1, 2, 8, 8).to(device), torch.zeros(1, 17).to(
    #     device),), "models/rating_ranges/Graphs/boards_mirrors.onnx", input_names=["Position", "Evaluation"], output_names=["Probabilities", "hn", "cn"])
