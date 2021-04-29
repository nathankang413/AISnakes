from utils.natural_selection import *

start_time = time.time()

starting_epsilon = 5

brains = rand_gen_brains(num_brains=100, sizes=(18, 10, 10, 3), epsilon=starting_epsilon)
print(natural_selection(start_brains=brains,
                        frac_keep=0.1,
                        frac_new=0.1,
                        mutation_chance=0.2,
                        mutation_size=1,
                        num_gens=100,
                        epsilon=starting_epsilon,
                        tests_per_brain=5,
                        show_game=False,
                        num_to_avg=10,
                        frame_worth=1,
                        apple_worth=50,
                        snake_hunger=100,
                        move_score=5,
                        apple_food=50,
                        away_punishment=2,
                        save_file='save_brains/new_brain.txt'
                        ))

# getting time elapsed
seconds_flt = time.time() - start_time
seconds_ttl = seconds_flt // 1
minutes = seconds_ttl // 60
seconds = seconds_ttl % 60
print(f'\n time elapsed: {minutes} minutes and {seconds} seconds.')
