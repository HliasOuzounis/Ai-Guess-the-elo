import torch

def convert_positions_to_tensors(dataset, func):
    positions = []
    elo = []
    
    for rating_range, games in dataset.items():
        for game in games:
            position_tensor, elo_tensor = convert_position(game, func)
            positions.append(position_tensor)
            elo.append(elo_tensor)
            
        print(f"{rating_range} done")
        
    return positions, elo

def convert_position(game, func):
    board = game.board()
    positions = []
    
    for move in game.mainline_moves():
        board_position_tensor = func(board.fen())
        positions.append(board_position_tensor)
        board.push(move)
        
    # Pad the game if it ends on white's turn to batch white's and black's analysis together later
    if len(positions) % 2:
        positions.append(torch.zeros_like(positions[0]))
    white_positions = torch.stack([position for position in positions[::2]])
    black_positions = torch.stack([position for position in positions[1::2]])
    
    # Encoding whose turn it is as a second channel
    white_positions = torch.stack([white_positions, torch.ones_like(white_positions)], dim=1)
    black_positions = torch.stack([black_positions, -torch.ones_like(black_positions)], dim=1)
    
    white_elo = int(game.headers["WhiteElo"])
    black_elo = int(game.headers["BlackElo"])
    
    return torch.stack((white_positions, black_positions)), torch.Tensor([white_elo, black_elo])

def fen_to_bitboard(fen_str):
    bitboard = torch.zeros(12, 64)
    fen, move, _castle, _en_passant, _halfmove, _fullmove = fen_str.split(" ")
    
    mappings = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
    }
    
    row, col = 0, 0
    for char in fen:
        if char == "/":
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            bitboard[mappings[char], row * 8 + col] = 1
            col += 1
    return bitboard

def fen_to_bitboard_mirror(fen_str):
    bitboard = torch.zeros(12, 64)
    fen, move, _castle, _en_passant, _halfmove, _fullmove = fen_str.split(" ")
    
    mappings = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
    }   if move == "w" else {
        "p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5,
        "P": 6, "N": 7, "B": 8, "R": 9, "Q": 10, "K": 11,
    }
    fen = fen if move == "w" else fen[::-1]
    row, col = 0, 0
    for char in fen:
        if char == "/":
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            bitboard[mappings[char], row * 8 + col] = 1
            col += 1
    return bitboard

def fen_to_board(fen_str):
    board = torch.zeros(8, 8)
    fen, move, _castle, _en_passant, _halfmove, _fullmove = fen_str.split(" ")
    normalizer = 20
    mappings = {
        "P": 1, "N": 3, "B": 3.5, "R": 5, "Q": 9, "K": 20,
        "p": -1, "n": -3, "b": -3.5, "r": -5, "q": -9, "k": -20,
    } 
    row, col = 0, 0
    for char in fen:
        if char == "/":
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            board[row, col] = mappings[char] / normalizer
            col += 1
    return board

def fen_to_board_mirror(fen_str):
    board = torch.zeros(8, 8)
    fen, move, _castle, _en_passant, _halfmove, _fullmove = fen_str.split(" ")
    normalizer = 20
    mappings = {
        "P": 1, "N": 3, "B": 3.5, "R": 5, "Q": 9, "K": 20,
        "p": -1, "n": -3, "b": -3.5, "r": -5, "q": -9, "k": -20,
    } if move == "w" else {
        "P": -1, "N": -3, "B": -3.5, "R": -5, "Q": -9, "K": -20,
        "p": 1, "n": 3, "b": 3.5, "r": 5, "q": 9, "k": 20,
    }
    fen = fen if move == "w" else fen[::-1]
    row, col = 0, 0
    for char in fen:
        if char == "/":
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            board[row, col] = mappings[char] / normalizer
            col += 1
            
    return board


if __name__ == "__main__":
    from elo_guesser.helper_functions.elo_range import get_rating_ranges
    import chess.pgn
    rating_ranges = get_rating_ranges()[3:-3].tolist()
    rating_ranges = tuple(map(tuple, rating_ranges))
    
    chess_games = {rating: [] for rating in rating_ranges}
    min_elo = 9999
    max_elo = 0

    games_per_rating = 1
    start_index = 0

    for rating_range in rating_ranges:
        start = start_index
        lower_bound = rating_range[0]
        upper_bound = rating_range[1]
        
        file = f"../datasets/outputs/{str(lower_bound)}-{str(upper_bound)}.pgn"


        with open(file) as f:
            while len(chess_games[rating_range]) < games_per_rating:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                if any(time_control in game.headers["Event"] for time_control in [
                    "Correspondence", "Daily", "Classical", "Bullet", "UltraBullet"
                ]):
                    continue
                if game.headers["WhiteElo"] == "?" or game.headers["BlackElo"] == "?":
                    continue
                if (
                    not lower_bound <= int(game.headers["WhiteElo"]) <= upper_bound
                    and not lower_bound <= int(game.headers["BlackElo"]) <= upper_bound
                ):
                    continue
                if not game.mainline_moves():
                    continue
                if len(list(game.mainline_moves())) < 15:
                    continue
                if start > 0:
                    start -= 1
                    continue
                
                chess_games[rating_range].append(game)
                
                min_elo = min(min_elo, int(game.headers["WhiteElo"]), int(game.headers["BlackElo"]))
                max_elo = max(max_elo, int(game.headers["WhiteElo"]), int(game.headers["BlackElo"]))
                
    position_converter_types = {
        "bitboards": fen_to_bitboard,
        "bitboards_mirrors": fen_to_bitboard_mirror,
        "boards": fen_to_board,
        "boards_mirrors": fen_to_board_mirror,
    }
    position_type = "boards_mirrors"

    positions, elo = convert_positions_to_tensors(chess_games, position_converter_types[position_type])
    
    print(len(positions))
    print(positions[0].shape)
