import game

if __name__ == '__main__':
    print("YAY LET'S START RUNNING STUFF")

    # PART1
    # game.play_game("human", "human")

    # PART2
    # playersArr = ["random","greedy", "greedy_improved", "minimax", "alpha_beta", "expectimax"]
    # resultArr=[]
    # for player1 in playersArr:
    #     for player2 in playersArr:
    #         if player1 == "minimax" or player1 == "alpha_beta" or player1 == "expectimax" or player2 == "minimax" or player2 == "alpha_beta" or player2 == "expectimax":
    #             resultArr.append(game.play_tournament(player1, player2, 3))
    #         else:

    #             resultArr.append(game.play_tournament(player1, player2, 50))
    # for item in resultArr:
    #     print(item)
    game.play_tournament("greedy_improved", "random", 500)
    #game.play_tournament("greedy_improved", "random", 50)
    #game.play_tournament("greedy", "random", 100)
