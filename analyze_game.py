import torch
import chess.engine

from tqdm import tqdm

def fen_to_bitboard(fen_str):
    mapings = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
    }   
    
    bitboard = torch.zeros(12, 64)
    fen, move, _castle, _en_passant, _halfmove, _fullmove = fen_str.split(" ")
    row, col = 0, 0
    for char in fen:
        if char == "/":
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            bitboard[mapings[char], row * 8 + col] = 1
            col += 1
    # Flatten the bitboard and add whose move it is
    return torch.cat((torch.tensor([1 if move == "w" else -1]), bitboard.flatten()))


def analyze_game(game, engine, device):
    mate_score = 9999
    board = game.board()
    analysis = []
    
    moves = list(game.mainline_moves())
    total_moves = len(moves)
    for i in tqdm(range(total_moves), desc="Analyzing game"):
        move = moves[i]
        pad_value = -9999 if board.turn else 9999
        
        board_position_tensor = fen_to_bitboard(board.fen())
        
        # top_moves = get_top_moves(board, nof_moves)
        # top_moves_tensor = pad_tensor(torch.tensor([move[1] for move in top_moves]), nof_moves, pad_value)
        best_move_tensor = torch.tensor([engine.analyse(board, chess.engine.Limit(time=0.1))["score"].white().score(mate_score=mate_score)])
        
        board.push(move)
        
        after_move_tensor = torch.tensor([engine.analyse(board, chess.engine.Limit(time=0.1))["score"].white().score(mate_score=mate_score)])

        # analysis.append(torch.cat((board_position_tensor, top_moves_tensor, after_move_tensor)))
        analysis.append(torch.cat((board_position_tensor, best_move_tensor, after_move_tensor)))
    
    # Pad the game if it ends on white's turn to batch white's and black's analysis together later
    if len(analysis) % 2:
        analysis.append(torch.zeros_like(analysis[0]))
    
    white_analysis = torch.stack([position for position in analysis[::2]])
    black_analysis = torch.stack([position for position in analysis[1::2]])
    
    return torch.stack((white_analysis, black_analysis)).to(device)