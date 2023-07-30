# Ai-Guess-the-Elo
An attempt at creating a neural network to guess a player's ELO rating based on their chess games/moves.

I had this idea in mind before chess.com released their version of an elo guessing ai but I postponed it long enough that they beat me to it. I was inspired by [Gothamchess' Guess the Elo series](https://www.youtube.com/watch?v=0baCL9wwJTA&list=PLBRObSmbZluRiGDWMKtOTJiLy3q0zIfd7).

All the effort on the project is condensed into a single file. You can run the `guess_the_elo.py` file with a pgn file as an argument. It will analyze the game, load the ai model and make a rating prediction for white and for black. For more info check [Usage instructions](#Usage).

## The Backbone
The project uses LSTM models from the [PyTorch](https://pytorch.org) library to make the elo predictions. The models are fed games analyzed by stockfish and the [python-chess](https://python-chess.readthedocs.io/en/latest/#) library.

LSTM models were used, firstly as a learning experience, and secondly because their "memory" feature I thought closely resembles how a human would analyze a game. For more explanations behind the decisions made read the [Decisions Explained](/elo_ai/models/Decisions_Explained.md) file. 

For the training data, games from all elo ranges from the [open lichess database June 2018](https://database.lichess.org/) were used after they were analyzed and modified accordingly. To speed up the process of uniformly selecting games of all elo ranges I used [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/).

## Results
The model trained, as explained in the [Decisions Explained](/elo_ai/models/Decisions_Explained.md) file predicts the probability of the elo being in one of 16 rating ranges.
```
<400, 400-600, 600-800, 800-1000, 1000-1200, 1200-1400, 1400-1600, 1600-1800, 1800-2000, 2000-2200, 2200-2400, 2400-2600, 2600-2800, 2800-3000, 3000-3200, 3200-3400
```
To find the true elo of the player, it finds the mean of the normal distribution with those probabilities per range.
### Training
The model was trained on 20000 games played on lichess.org in June 2018, 2000 games of each elo range were selected. From those, 15% was used for testing and from the remaining 85%, 10% was used for validation. It was trained for 20 epochs, enough to decrease the loss while also avoiding overfitting.

#### Rating Ranges model
For the model Cross Validation was used as the loss function. We can think of it as a classification problem with 16 classes, the rating ranges.


<p align="center">
  <img src="/elo_ai/models/rating_ranges/Graphs/loss_plot.png" alt="training loss rating ranges model">
</p>


It is important to note that a random classification model with 16 classes would have an average Cross Validation loss of `ln(16) = 2.772`. That means the rating ranges model is significantly better than a random one.


We can clearly see a downwards trend that plateaus. We have reached a stagnation in training while avoiding overfitting.


### Predictions


To rate the accuracy of the model it is tested on 3000 games (= 6000 predictions, 2 per game for white and black) and pitted against two other trivial models. A random guessing one and one that always predicts a rating around the middle of the rating ladder (800 - 3000). We also give the models a leeway of 200 points on their guess. As the leeway increases so do the correct guesses but the precision is lowered.


#### Trivial models


The 2 trivial models seem to have around a 17-23% accuracy for a 200 point leeway and the average difference between the real value and the prediction is 500 points for the constant model and 700 for the random model. The constant models edges out the random one by a bit, both in accuracy and in average. Both of them are very bad at guessing the real elo of a player.


#### Rating Ranges model


Plotting the real values with the predictions we can visualize the accuracy of the model.


<p align="center">
  <img src="/elo_ai/models/rating_ranges/Graphs/predictions.png" alt="predictions rating ranges model">
</p>


On this graph we can see 2000 of the 6000 predictions made.  The closer to the red line (perfect matching) a dot is, the better the prediction. Points that fall between the two green lines have a difference of less than 200 points. There we find around 55% of the data. Additionally, 90% of the predictions are between the two purple lines, 500 points difference. This makes sense considering a bad game by a 1700 player could resemble a game of a 1200 player. Conversely, a good game by a 1200 could be mistaken for a 1700.


The points follow the lines pretty closely meaning the model has understood the differences between a good and a not so good player and can make predictions accordingly. It's clear though that the model has some troubles predicting low elo games and very high elo games. For games <1000 elo, the model tends to overestimate the player and for games >2600 it tends to underestimate them. The tradeoff is a good modeling of the middle of the rating ladder.


Considering the variance of a players strength, these results are pretty good.

These results can be found in the [jupyter notebook](/elo_ai/models/rating_ranges/lstm_train_rating_ranges.ipynb) that was used.


## Usage
To get elo predictions for your chess games clone this repository and install the required libraries
```
git clone https://github.com/HliasOuzounis/Ai-Guess-the-elo
pip install -r requirements.txt
```
or download the files from the release page (outdated: old models, less accurate), 


and run the `guess_the_elo.py` file with python
```
python guess_the_elo.py [--engine engine-dir] [-c] pgn_file
```

Because the training dataset was from lichess.org, the model has learned to predict lichess ratings. If your game is from chess.com pass the -c flag so the elo gets converted. On average chess.com ratings are 400 points lower than lichess.org.


Because the games need to be analyzed by an engine first, you need to have a chess engine installed, preferably [stockfish](https://stockfishchess.org/download/). Pass the path as the --engine argument when calling the file. On linux you can find the installed path of stockfish with `which stockfish`. On my arch-based distro that was `/usr/bin/stockfish` which I have used as the default.


Finally, pass the pgn file which contains the game as the last argument. It needs to be parsable by chess.pgn.read_game(). If you copy the pgn from the website's "share" feature onto a plain text file it should be good enough.

### Examples


Feeding the models [one of my games](https://lichess.org/bNLqqjHP/black#0)

```
python guess_the_elo.py my_game.pgn
```
<p align="center"> 
  <img src="datasets/showcase_games/game2.gif" alt="game2">
</p>
I get predictions:

- 2050 for white
- 2050 for black


Considering I'm around 2000 rated on Lichess, those predictions are very good.


For [other games](https://lichess.org/BoxuoUjy/black#0) the predictions aren't as accurate.

<p align="center">
  <img src="datasets/showcase_games/game1.gif" alt="game1">
</p>

The model predicts:

- 1500 for white
- 1300 for black

That's not really close to both of our ratings. It's possible though that we didn't play that well and the ratings it predicts are justified. It's hard to judge without an experienced player analyzing the games as well. 

### Additional Notes


We should keep in mind that guessing a player's rating off of a single game is not a very good metric since players can have good or bad games. 

Additionally, a player can't show his full level if their opponent plays badly and hands him the win. That means a player's elo prediction is indirectly affected by his opponent. (though the models judge a player solely on his moves, the positions that arise, which are determined by both players, are also taken into account)

Also, those ratings depend a lot on the stockfish evaluation of each game which isn't totally consistent even when analyzing the same game.

## Conclusions
Wow, chess is a very complex game, who would have thought! 

It seems the model was able to somewhat understand what it means to play at a higher level, but the training dataset was small considering the amount of chess games possible and the model not deep enough to perfectly grasp the level of a player based on their moves. It's also true that a players strength is difficult to measure based on just one game as the level of play has a lot of variance. Despite that, the accuracy that was achieved was satisfactory.

But it's safe to say that an experienced chess player/coach would probably make more accurate predictions than this model. Still, it was a fun project and a learning experience.