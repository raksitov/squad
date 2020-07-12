# Exploring speed and memory trade-offs for achieving optimum performance on SQuAD dataset

The code and the report for Stanford's CS224n 
(["Natural Language Processing with Deep Learning"](http://web.stanford.edu/class/cs224n/))
default final project. 

In this project I am building deep learning system for reading comprehension on Stanford
Question Answering Dataset (SQuAD). The code is adapted in part from the
[Neural Language Correction](https://github.com/stanfordmlgroup/nlc/) code by the Stanford
Machine Learning Group.

I trained a single model that achieved **76.37** F1 and **66.00** EM on dev dataset and
ensemble of 12 models that achieved **80.044** F1 and **71.971** EM on test dataset. The
ensemble model got 6th place (out of 100+ teams / submossions) on the class leaderboard
(see _TestLeaderboard.jpg_ for the snapshot of the leaderboard)
