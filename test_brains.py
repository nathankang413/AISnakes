from utils.train_ai_game import TrainAI
from utils.genetic_algo import *

# brain1 = SnakeBrain(file_name='save_brains/brain1.txt')
brain1 = SnakeBrain(file_name='save_brains/from_og_1.01.txt')

# NORMAL GAME
# game = TrainAI(frame_time=0.1,
#                starting_length=5,
#                num_apples=3,
#                show_game=True,
#                num_snakes=2,
#                frame_worth=0,
#                apple_worth=1,
#                max_frames=10000,
#                snake_hunger=10000,
#                move_score=0,
#                brains=[brain1, brain2],
#                apple_food=100,
#                away_punishment=0)

# TRAINING CONDITIONS GAME
game = TrainAI(frame_time=0.05,
               starting_length=5,
               num_apples=1,
               show_game=True,
               num_snakes=1,
               frame_worth=1,
               max_frames=10000,
               brains=[brain1],
               apple_worth=50,
               snake_hunger=100,
               move_score=5,
               apple_food=50,
               away_punishment=2,)

game.run_game()
