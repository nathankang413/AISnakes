from utils.natural_selection import *

start_time = time.time()

brain1 = SnakeBrain(file_name='save_brains/from_new1.txt')
brains = [brain1]
for i in range(99):
    brains.append(brain1.copy_and_mutate(0.2, 1))

print(natural_selection(start_brains=brains,
                        frac_keep=0.1,
                        frac_new=0.1,
                        mutation_chance=0.2,
                        mutation_size=1,
                        num_gens=50,
                        epsilon=5,
                        tests_per_brain=7,
                        show_game=False,
                        num_to_avg=10,
                        frame_worth=1,
                        apple_worth=50,
                        snake_hunger=100,
                        move_score=5,
                        apple_food=50,
                        away_punishment=2,
                        save_file='save_brains/from_new1.txt',
                        graduated_frames=False,
                        max_frames=10000,
                        ))

# getting time elapsed
seconds_flt = time.time() - start_time
seconds_ttl = seconds_flt // 1
minutes = seconds_ttl // 60
seconds = seconds_ttl % 60
print(f'\n time elapsed: {minutes} minutes and {seconds} seconds.')
