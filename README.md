# playing_card_detection

I generated images containing playing cards and trained using [Yolov7](https://github.com/WongKinYiu/yolov7).
I borrowed some ideas of data generation from [geaxgx](https://github.com/geaxgx/playing-card-detection).

## How to use this repo


## Requirements

- pytorch
- yolov7

## Data generation

In this project, I used the cards used in WSOP Poker, as shown below.
Note that the red is shown in blue.
![Cards](figures/cards.png)

First I detected the region of interest in each card, namely the rank and suit at the corner of the card. 

