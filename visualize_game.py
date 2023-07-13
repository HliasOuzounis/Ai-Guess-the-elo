import chess.pgn
import chess.svg
import chess.engine

import  matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import cairosvg
import time

from models.lstm_network import initialize_model
from analyze_game import analyze_game, convert_position, fen_to_board_mirror
from guess_the_elo import guess_elo_range

def create_window():
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    return fig, axs
    

def game_visualized(game):
    window, axs = create_window()
    
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        board_image = Image.open(BytesIO(cairosvg.svg2png(chess.svg.board(board=board))))
        axs[0].imshow(board_image)
        time.sleep(1)
        plt.show()
        
def main():
    import torch
    game = chess.pgn.read_game(open("my_game2.pgn"))
    # game_visualized(game)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 82
    hidden_size = 128
    num_layers = 2
    learning_rate = 0.001
    lstm_model, _optimizer = initialize_model(input_size, hidden_size, num_layers, device, learning_rate, num_classes=10)
    lstm_model.load_state_dict(
        torch.load("models/rating_ranges/boards_mirrors.pt")
    )
    engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")
    white_analysis, black_analysis = analyze_game(game, engine, 99999)
    white_positions, black_positions = convert_position(game, fen_to_board_mirror)
    
    white_analysis = torch.cat((white_positions, white_analysis), dim=-1)
    black_analysis = torch.cat((black_positions, black_analysis), dim=-1)
    
    game = chess.pgn.read_game(open("my_game1.pgn"))
    moves = list(game.mainline_moves())
    board = game.board()
    h_w, c_w, h_b, c_b = None, None, None, None

    rating_ranges = [
        "1000", "1200", "1400", "1600", "1800",
        "2000", "2200", "2400", "2600", "3400"
    ]
    print(white_analysis.shape)
    for i in range(len(moves)//2):
        white_predictions, (h_w, c_w) = lstm_model(white_analysis[i].unsqueeze(0).to(device), h_w, c_w, False)
        board.push(moves[2*i])
        black_predictions, (h_b, c_b) = lstm_model(black_analysis[i].unsqueeze(0).to(device), h_b, c_b, False)
        board.push(moves[2*i+1])
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f"Move {i+1}")
        axs[0].bar(rating_ranges, white_predictions[0].cpu().detach().numpy())
        axs[0].set_title(f"White {guess_elo_range(white_predictions[0])}")
        axs[1].bar(rating_ranges, black_predictions[0].cpu().detach().numpy())
        axs[1].set_title(f"Black {guess_elo_range(black_predictions[0])}")
        plt.show()
    white_predictions, (h_w, c_w) = lstm_model(white_analysis.to(device), None, None, False)
    black_predictions, (h_b, c_b) = lstm_model(black_analysis.to(device), None, None, False)
    print(guess_elo_range(white_predictions[0]))
    print(guess_elo_range(black_predictions[0]))
    
    engine.close()
    
if __name__ == "__main__":
    main()