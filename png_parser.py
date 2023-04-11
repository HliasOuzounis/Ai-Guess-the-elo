import chess.pgn
import stockfish

engine = stockfish.Stockfish(parameters={"Threads": 2})

games = []
with open("lichess_db_standard_rated_2013-01.pgn") as f:
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        games.append(game)
        if len(games) > 1:
            break

game = games[0]

engine.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print(engine.get_top_moves(10))