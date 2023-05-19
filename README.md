# Ai-Guess-the-Elo

This is the release branch without the dataset, the model training and all the explanations.

## Usage
To get elo predictions for your chess games clone this reporitory after you have installed the necessary python libraries
```
pip install torch tqdm python-chess
git clone https://github.com/HliasOuzounis/Ai-Guess-the-elo
```
or download the files from the release page, 


and run the `guess_the_elo.py` file with python
```
python guess_the_elo.py [--engine engine-dir] [-c] pgn_file
```

Because the training dataset was from lichess.org, the model has learned to predict lichess ratings. If your game is from chess.com pass the -c flag so the elo gets converted. On average chess.com ratings are 400 points lower than lichess.org.


Because the games need to be analyzed by an engine first, you need to have a chess engine installed, preferably [stockfish](https://stockfishchess.org/download/). Pass the path as the --engine argument when calling the file. On linux you can find the installed path of stockfish with `which stockfish`. On my arch-based distro that was `/usr/bin/stockfish` which I have used as the default.


Finally, pass the pgn file which contains the game as the last argument. It need to be parsable by chess.pgn.read_game.
If you copy the pgn from the website's "share" feature onto a plain text file it should be good enough.