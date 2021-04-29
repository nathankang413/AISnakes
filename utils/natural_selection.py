from utils.train_ai_game import TrainAI
from utils.genetic_algo import *


def test_brains(brains, tests_per_brain=1, max_frames=10000, show_game=True, frame_worth=1, apple_worth=50,
                snake_hunger=100, move_score=5, apple_food=50, away_punishment=2):
    scores = []

    # iterate through brains to get a score for each
    for b in range(len(brains)):

        snake_scores = []

        # run game tests_per_brain times

        for _ in range(tests_per_brain):
            game = TrainAI(frame_time=0.0,
                           starting_length=10,
                           num_apples=1,
                           show_game=show_game,
                           num_snakes=1,
                           frame_worth=frame_worth,
                           apple_worth=apple_worth,
                           max_frames=max_frames,
                           snake_hunger=snake_hunger,
                           move_score=move_score,
                           brains=[brains[b]],
                           apple_food=apple_food,
                           away_punishment=away_punishment
                           )

            game.run_game()
            snake_scores.append(game.snakes[0].score)

        # use median score
        scores.append(sorted(snake_scores)[int(tests_per_brain / 2)])

    return scores


def natural_selection(start_brains=rand_gen_brains(10, (18, 10, 10, 3), 1), frac_keep=0.1, frac_new=0.1,
                      mutation_chance=0.5, mutation_size=0.25, num_gens=10, epsilon=1, tests_per_brain=1,
                      show_game=True, num_to_avg=10, frame_worth=1, apple_worth=50, snake_hunger=100,
                      move_score=5, apple_food=50, away_punishment=2, save_file='current_brain.txt',
                      graduated_frames=True, max_frames=10000):

    num_brains = len(start_brains)
    brains = start_brains
    scores = []
    rolling_averages = []
    num_new = int(num_brains * frac_new)

    print(f'num brains = {num_brains}')
    print(f'Epsilon = {epsilon}')
    print(f'Frac_keep = {frac_keep}')
    print(f'Frac_new = {frac_new}')
    print(f'mutation chance = {mutation_chance}')
    print(f'mutation size = {mutation_size}')
    print(f'num_gens = {num_gens}')
    print(f'tests_per_brain = {tests_per_brain}')

    top_scores = []
    for gen in range(num_gens):
        frames = max_frames
        if graduated_frames:
            frames = (gen+1)*50

        scores = test_brains(brains, tests_per_brain=tests_per_brain, max_frames=frames, show_game=show_game,
                             frame_worth=frame_worth, apple_worth=apple_worth, snake_hunger=snake_hunger,
                             move_score=move_score, apple_food=apple_food, away_punishment=away_punishment)
        top_brains = get_top_brains(brains, scores, frac_keep)

        # save top brain
        best_brain = get_top_brains(brains, scores, 0.00001)[0]
        best_brain.save_brain(save_file)

        brains = top_brains

        # fill the number of new brains to avoid local maxima
        brains += rand_gen_brains(num_new, sizes=(18, 10, 10, 3), epsilon=epsilon)

        # fill remaining generation with offspring from previous bests
        i = 0
        while len(brains) < num_brains:
            ind = i % len(top_brains)

            brains.append(top_brains[ind].copy_and_mutate(mut_chance=mutation_chance, mut_size=mutation_size))
            i += 1

        # store and show data
        top_scores.append(max(scores))

        num_avg = num_to_avg
        if gen >= num_avg - 1:
            rolling_averages.append(sum(top_scores[-num_avg:]) / num_avg)
        else:
            rolling_averages.append(sum(top_scores) / len(top_scores))
        print(f'Generation {gen}: {max(scores)}, rolling_average ({num_avg}): {rolling_averages[-1]}')

    plt.plot(top_scores)
    plt.plot(rolling_averages)
    plt.show()
    return get_top_brains(brains, scores, 0.00001)[0]



