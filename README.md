# How to Play
1. Run `game.py`.
2. Start the game by pressing the 's' key.
3. Show your hand sign to the camera.
4. The AI will also choose a hand sign.
5. The winner will be displayed on the screen.
6. To end the game, press the 'q' key.

# Adding Your Own Training Data
1. Activate training mode by pressing the 't' key.
2. Select a training slot (0-9) by pressing the corresponding number key.
    - Training slots 0-4 correspond to rock, paper, scissors, lizard, and spock respectively.
    - Additional training slots are included, allowing further extension of the game to train more hand signs.
3. Show your hand sign to the camera and press 'c' to capture the data.
4. The captured data will be saved to `./model/training/training_data.csv`.
5. Execute `handSignClassifier.ipynb` to train the model with the captured data.
6. Switch training slots after training a hand sign.