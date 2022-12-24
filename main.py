import game

if __name__ == '__main__':
    print("YAY LET'S START RUNNING STUFF")

    # PART1
    #game.play_game("human", "human")

    # PART2
    #game.play_tournament("minimax", "random", 50)
    #game.play_tournament("minimax", "greedy", 50)
    game.play_tournament("alpha_beta", "greedy_improved", 50)
