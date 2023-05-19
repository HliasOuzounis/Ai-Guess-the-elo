## Why LSTM?
This is the major decision that everything else in the project builds on.

Since the model will be used to analyze chess games with an arbitrary amount of moves we need a model that can handle sequential inputs. Unfortunately, RNNs, the simplest recurrent networks suffer a lot from the vanishing (or exploding) gradient and as such have fall victim to the long-term dependancy problem. Great moves or blunders in the earlier stages of the game would have little to no impact if the game drags on.

Then LSTM models are the logical continuation. They combat the long-term dependancy problem with a gating mechanic. Depending on the new input, the model decides whether or not to forget the previous data. 

This to me sounded a lot like how a human would approach guessing the elo of a chess game. Yes, maybe the player blundered in the beggining but made a comeback by playing a lot of good moves in a row. Then we could "forgive" the earlier mistake. Or the opposite. The player made a high level move but then immediatly blundered and lost all his advantage. Then we could consider the good move a fluke and ignore it.

There was a thought to use transformers considering their popularity recently and the fact that they can use attention to better understand the relations between data. If trained well they could look at games not in terms of singular moves but by whole ideas and plans, looking at 5 or 6 moves at a time. Though they are computationally more expansive and need a fixed input size. But that's something to consider for a future project.


## What's the input to the models?
The input to the modles is a 771 dimensional vector. It is a bitboard representation of the current position, meaning an 8x8=64 board for each piece for both colors (6x2=12) with a value of 1 if the squared is occupied by that type of piece or 0 if it's not. Additionaly it has a number to indicate whose turn it is and the evalutaion of stockfish (given 0.1s thinking time) before and after the move. 

I was hoping that the models would be able to recognize the centipawn loss as the difference before and after the move was played and base their prediction on that. Also, by having the board representation they could also calculate a "complexity score" of the position. If the position was very complex, they would be more forgiving and not be too harsh on mistakes.

That seemed to kinda work. I also tested the models without the board representation and they were noticably worse. To improve on that concept I would have liked to not only feed the models the stockfish evaluation before the player's move but also the scores of the top 5 or 10 moves of the engine. That way the model can figure can figure out how many good moves there are. If there are a lot of good moves and the player plays a bad one (based on centipawn loss) he should be punished more.

Unfortunately, processing power doesn't grow on trees. It already took my laptop 10 hours to process all the games. To add the top 10 moves would require more than 10x the time. I think this idea has some merit but I couldn't test it sadly.

## What's the output?
My first thought on the output of the model was a single value to predict the elo. This value was normalized to 3000 because neural networks work better with values from 0-1. The output layer is a softmax function to match this normalization. Additionally it smooths out very big or very small guesses so there aren't any crazy outliers. This is how the [single-output model](lstm_train_single_output.ipynb) works.

To improve on this approach I examined how the ELO rating system works. In general, a player's strength isn't simply a fixed value but has some variance from game to game. So, by modeling the player's strength as a probapility distribution over the whole elo rating range we can better model how they might play on a game to game basis. Some games they might play better than their elo, some worse, but it all averages out to their true elo rating. 

With that in mind I picked 10 elo rating ranges [<900, 900-1100, 1100-1300, ..., 2300-2500, >2500] and for each player in the training dataset I calculated the probapility of playing in each range by taking the integral of the probability density function in that range. The probability density function for each player is a normal distribution with mean their true elo rating and standard deviation 100. 

The output of this new model would be a 10-vector with the probabilities of the player being in each of the 10 ranges. To take these probabilities and calculate the mean, we take a weighted average over the ranges. For example if the output of the model was [0, 0, 0, 0, 0, 0.3, 0.4, 0.2, 0.1, 0], the true elo would be 0.3 * (1700 + 1900)/2 + 0.4 * (1900 + 2100)/2 + 0.2 * (2100 + 2300)/2 + 0.1 * (2300 + 2500)/2 = 2020. This has now turned into a classification problem (almost) where the model tries to predict the probability of the player's elo belonging to each range. That's the mechanism behind the [rating-ranges model](lstm_train_rating_ranges.ipynb).