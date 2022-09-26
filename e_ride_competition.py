# WORKING
# by Darko Radakovic
# 09/20/2022
# Montclair State University
# version 7 (Github ready)

import numpy as np
import random    # build in
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import statistics
from itertools import permutations  # build in
import argparse    # build in
import pickle    # build in
import os     # build in
from pathlib import Path    # build in
import sys    # build in
import datetime    # build in


# STRATEGIES OVERVIEW
# [Strategy 1] 'basic' [For player 1] find cell with most passengers, regardless their distance, chooses first passenger from list in this cell (old taxis without app strategy, by going to busy areas)
# [Strategy 2] 'low' [For player 2] find closest passenger lowest distance, disregarding profit
# [Strategy 3] 'passenger'[For player 1] find longest passenger start end (highest profit)
# [Strategy 4] 'distance_ratio' [For player 2] choosing score of passengers over battery
# [Strategy 5] 'battery' [For player 2] This method explores a passenger combination of the highest value and the third highest value (assuming that the second highest value will be chosen by the other player in the next turn), it decides whether to go first to the highest value or to the third highest values for the most optimal route.
# [Strategy 6] 'passenger2' [For player 2] always choosing highest score, then look at closest if more than one  -> BEST RESULTS



# Instantiate the parser
parser = argparse.ArgumentParser(description='MiniMax E-Ride sharing game between two players, playing with different'
        ' strategies. One can manipulate the number of passengers, the change in passenger numbers, the grid size, ' \
    'player strategy type, player starting location, the number of simulations and to enable MiniMax.')
parser.add_argument('--ptot', type=int, default=100, help='passenger number') # number of passengers  (minimum ±50)
parser.add_argument('--size', type=int, default=[5,10], help='grid size, list input of two values')   # size of matrix  (x by y)
parser.add_argument('--change', type=str, default='increasing', help='passengers numbers change "falling", "stable" or "increasing"')   # chose passengers numbers change: 'falling', 'stable', 'increasing'
parser.add_argument('--minimax', type=int, default=1, help='minimax algorithm for player2, 1 if true, 0 if false')   # decide if minimax algorithm is turned on for player2
parser.add_argument('--strategy1', type=int, default=3, help='strategy for player 1, "passenger" or "basic"')   #  choose from: 'passenger' or 'basic'
parser.add_argument('--strategy2', type=int, default=6, help='strategy for player 2, "low", "distance_ratio", "battery" or "passenger2"')   # choose from: 'low', 'distance_ratio', 'battery' or 'passenger2'
parser.add_argument('--last_loc1', type=int, default=[0,0], help='Player1 starting location, list input of two values')   # Player1 starting location
parser.add_argument('--last_loc2', type=int, default=[0,0], help='Player2 starting location, list input of two values')   # Player2 starting location
parser.add_argument('--sim', type=int, default=100, help='number of simulations')   # Simulation total
parser.add_argument('--save', action='store_true', help='save results to pickle file')
args = parser.parse_args()

## PARAMETERS Can be set with flag
ptot = args.ptot  # number of passengers  (minimum ±50)
size = args.size
change = args.change  # chose passengers numbers change: 'falling', 'stable', 'increasing'
minimax = args.minimax   # decide if minimax algorithm is turned on for player2
strategy1 = args.strategy1 #  choose from: 'passenger' or 'basic'
strategy2 = args.strategy2  # choose from: 'low', 'distance_ratio', 'battery'
last_loc1 = args.last_loc1  # Car1 starting location, multiple players
last_loc2 = args.last_loc2  # Car2 starting location, multiple players
sim = args.sim       # amount of simulations
save_output = args.save  # save results


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# used as suffix in file name when saving results, based local date and time
date = datetime.datetime.now().strftime("%x").replace('/','') + "_" + datetime.datetime.now().strftime("%X").replace(':','.')  # getting date for version

# --------------------------------------------------
# ---------------MANUALLY FILL IN PARAMETERS---------------
# -----Matrix size, passenger amount, and start locations---
# --------------------------------------------------
count = 0  # start time (no need to change)


# 3 passengers with start, stop coordinates, based on image of test 1 seen above.
# Algorithm takes the first coordinates as first passenger, etc.
# passengers = [ [[2,0],[2,2],1], [[0,1],[0,2],2], [[1,2],[2,0],2] ]
# passengers = [ [[3,2],[2,0],2], [[0,2],[1,1],1], [[3,2],[6,0],3], [[4,0],[9,2],5] ]
# passengers = [ [[2,0],[2,2]], [[0,1],[0,2]], [[1,2],[2,0]] ]






# --------------------------------------------------
# ---------------AUTO---------------------
# --------------------------------------------------
def build_matrix(ptot,size):
    passengers = []
    for i in range(0, ptot):
        p1, p2 = 0, 0
        p1 = [random.randint(0, size[1] - 1), random.randint(0, size[0] - 1)]  # start position x-axis,y-axis
        p2 = [random.choice(list(filter(lambda ele: ele != p1[0], range(0, size[1] - 1)))),
              random.randint(0, size[0] - 1)]  # end position x-axis,y-axis, except already chosen start location
        # p2 = [random.randint(0, size[0]),random.randint(0, size[0])] # end position x-axis,y-axis, not accounting for start location
        passengers.append([p1, p2])

    # passengers2 = passengers.copy()  # create a copy
    # np.array(passengers)[:,0]  # shows only

    ### MATRIX
    # Create MATRIX board with starting locations
    matrix = np.zeros((size[1], size[0]))
    for i in range(0, size[1]):
        for j in range(0, size[0]):
            for p in range(len(passengers)):
                if passengers[p][0] == [i, j]:
                    matrix[i, j] += 1

    # print(sum(sum(matrix))) # Amount of passengers

    # Calculate Density of nearby passengers +1 cells
    density_matrix = np.zeros((size[1], size[0]))
    for i in range(0, size[1]):
        for j in range(0, size[0]):
            d = 0  # distance value
            d += matrix[i, j]
            try:
                d += matrix[i + 1, j]
            except:
                d += 0
            try:
                d += matrix[i - 1, j]
            except:
                d += 0
            try:
                d += matrix[i, j + 1]
            except:
                d += 0
            try:
                d += matrix[i, j - 1]
            except:
                d += 0
            density_matrix[i, j] = d

    return matrix, density_matrix, passengers

def add_passengers(passengers,size, mode):
    if mode == 'falling':
        add = random.randint(0, 0)  # adding passengers random up to 2 new ones, keeps the passengers somehow equal
    if mode == 'stable':
        add = random.randint(0, 2)  # adding passengers random up to 2 new ones, keeps the passengers somehow equal
    if mode == 'increasing':
        add = random.randint(2, 5)  # adding passengers random up to 2 new ones
    new_p = []  # store start location of new p
    for i in range(0, add):
        p1, p2 = 0, 0
        p1 = [random.randint(0, size[1] - 1), random.randint(0, size[0] - 1)]  # start position x-axis,y-axis
        p2 = [random.choice(list(filter(lambda ele: ele != p1[0], range(0, size[1] - 1)))),
              random.randint(0, size[0] - 1)]  # end position x-axis,y-axis, except already chosen start location
        # p2 = [random.randint(0, size[0]),random.randint(0, size[0])] # end position x-axis,y-axis, not accounting for start location
        new_p.append(p1)
        passengers.append([p1, p2])
    if add == 0:
        new_p =[0,0]
    return passengers, new_p

# --------------------------------------------------
def eridegame(matrix, passengers, last_loc, new_p):
    ### MATRIX
    # Update Matrix with newly added passengers
    try:  # see if no error if new_p is [0,0]
        if new_p[0][0] + new_p[0][1] > 0:
            try:
                len(new_p[1]) > 0  # check if new_p has two passengers
                for i in range(len(new_p)):
                    matrix[new_p[i][0], new_p[i][1]] += 1  # add one passenger to cell
            except:
                for i in range(len(new_p)):
                    matrix[new_p[i][0], new_p[i][1]] += 1  # add one passenger to cell
    except:
        if new_p[0] + new_p[1] > 0:  # check if new_p is not [0,0]
            matrix[new_p[0], new_p[1]] += 1   # add one passenger to cell

    # Update Density of nearby passengers +1 cells
    density_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            d = 0  # distance value
            d += matrix[i, j]
            try:
                d += matrix[i + 1, j]
            except:
                d += 0
            try:
                d += matrix[i - 1, j]
            except:
                d += 0
            try:
                d += matrix[i, j + 1]
            except:
                d += 0
            try:
                d += matrix[i, j - 1]
            except:
                d += 0
            density_matrix[i, j] = d

    # ## Battery cost (distance from last location + distance end and start passenger)
    # battery_p = []
    # for i in range(len(passengers)):
    #     v = 0
    #     v += abs(last_loc[0] - passengers[i][0][0])  # distance from last location to first passenger in comb
    #     v += abs(last_loc[1] - passengers[i][0][1])  # distance from last location to first passenger in comb
    #     v += abs(passengers[i][0][0] - passengers[i][1][0])  # distance end and start passenger of comb in order
    #     v += abs(passengers[i][0][1] - passengers[i][1][1])  # distance end and start passenger of comb in order
    #     battery_p.append(v)

    battery_to_p = []   # distance to passenger
    battery_ps_pe = []   # distance passenger start to passenger end location
    for i in range(len(passengers)):
        v = 0
        v2 = 0
        v += abs(last_loc[0] - passengers[i][0][0])  # distance from last location to first passenger in comb
        v += abs(last_loc[1] - passengers[i][0][1])  # distance from last location to first passenger in comb

        v2 += abs(passengers[i][0][0] - passengers[i][1][0])  # distance end and start passenger of comb in order
        v2 += abs(passengers[i][0][1] - passengers[i][1][1])  # distance end and start passenger of comb in order
        battery_to_p.append(v)
        battery_ps_pe.append(v2)

    return matrix, density_matrix, battery_to_p,battery_ps_pe

#-----------------------------------------#
### ------------ MINIMAX ---------------###
def minimax_algorithm(matrix, passengers, battery_to_p, battery_ps_pe, last_loc, player, strategy1, strategy2, minimax):

    if player == 1:
        ### HIGHEST PASSENGER DISTANCE  [STANDARD MAX. PROFIT STRATEGY, heuristic]
        # Strategy 3
        if strategy1 == 3:  # choosing score of passengers over battery
            pass_remove1 = battery_ps_pe.index(max(battery_ps_pe))  # find longest passenger start end (highest profit)
            loc1 = []
            score1 = 0
            battery1 = 0
            battery1 = battery_to_p[pass_remove1] + battery_ps_pe[pass_remove1]

            score1 = battery_ps_pe[pass_remove1]  # profit = distance of passenger start to end
            loc1 = passengers[pass_remove1][0]
            # matrix[options[0][0], options[1][0]] = 0  # remove passenger
            # last_loc1 = [options[0][0], options[1][0]]

            score_fin = score1
            battery_fin = battery1
            loc_fin = loc1
            passengers.remove(passengers[pass_remove1])  # remove passenger from list

        # Strategy 1
        if strategy1 == 1:  # find cell with most passengers, regardless there distance, chooses first passenger from list in this cell (OLD strategy without heuristic)
            ### HIGHEST MATRIX CELL

            ## Battery cost (distance from last location + distance end and start passenger)
            battery_p = []
            for i in range(len(passengers)):
                v = 0
                v += abs(last_loc[0] - passengers[i][0][0])  # distance from last location to first passenger in comb
                v += abs(last_loc[1] - passengers[i][0][1])  # distance from last location to first passenger in comb
                v += abs(passengers[i][0][0] - passengers[i][1][0])  # distance end and start passenger of comb in order
                v += abs(passengers[i][0][1] - passengers[i][1][1])  # distance end and start passenger of comb in order
                battery_p.append(v)

            loc1 = []
            loc1 = np.where(matrix == matrix.max())  # find cells with max amount of passengers
            # options = np.where(density_matrix == density_matrix.max())  # find cells with max density.
            score1 = 0
            score1 = matrix.max()
            battery1 = 0
            for i in range(len(passengers)):
                if passengers[i][0] == [loc1[0][0], loc1[1][0]]:
                    battery1 = battery_p[i]
                    pass_remove1 = i  # index of chosen passenger (to be removed after picked up)
            loc1 = [loc1[0][0], loc1[1][0]]
            # matrix[options[0][0], options[1][0]] = 0  # remove passenger
            # last_loc1 = [options[0][0], options[1][0]]
            score_fin = score1
            battery_fin = battery1
            loc_fin = loc1
            passengers.remove(passengers[pass_remove1])  # remove passenger from list

    if player == 2:
        # Strategy 2
        if strategy2 == 2:  # find cell lowest distance

            ## Battery cost (distance from last location + distance end and start passenger)
            battery_p = []
            for i in range(len(passengers)):
                v = 0
                v += abs(last_loc[0] - passengers[i][0][0])  # distance from last location to first passenger in comb
                v += abs(last_loc[1] - passengers[i][0][1])  # distance from last location to first passenger in comb
                v += abs(passengers[i][0][0] - passengers[i][1][0])  # distance end and start passenger of comb in order
                v += abs(passengers[i][0][1] - passengers[i][1][1])  # distance end and start passenger of comb in order
                battery_p.append(v)

            # MINIMAX
            ### BATTERY CALCULATION, passenger with lowest battery demand
            if minimax == 1:  # if minimax is turned on
                # if the distance to the closest passenger over distance to passenger multiplied by 1.5 is larger than the max profit possible
                if min(battery_p) * 2.5 > max(battery_ps_pe):  # 1.05 is most optimal
                    pass_remove2 = battery_p.index(min(battery_p))  # choose the highest profit/distance passenger
                    battery2 = 0
                    battery2 = battery_to_p[pass_remove2] + battery_ps_pe[pass_remove2]
                    loc2 = []
                    loc2 = passengers[pass_remove2][0]
                    score2 = 0
                    score2 = battery_ps_pe[pass_remove2]  # profit = distance of passenger start to end
                else:
                    loc2 = np.where(matrix == matrix.max())  # find cells with max amount of passengers
                    score2 = 0
                    score2 = matrix.max()
                    battery2 = 0
                    for i in range(len(passengers)):
                        if passengers[i][0] == [loc2[0][0], loc2[1][0]]:
                            battery2 = battery_p[i]
                            pass_remove2 = i  # index of chosen passenger (to be removed after picked up)
                    loc2 = [loc2[0][0], loc2[1][0]]
            else:
                pass_remove2 = battery_p.index(min(battery_p))  # Always choose this strategy, even if lower
                battery2 = 0
                battery2 = battery_to_p[pass_remove2] + battery_ps_pe[pass_remove2]
                loc2 = []
                loc2 = passengers[pass_remove2][0]
                score2 = 0
                score2 = battery_ps_pe[pass_remove2]  # profit = distance of passenger start to end
            score_fin = score2
            battery_fin = battery2
            loc_fin = loc2
            passengers.remove(passengers[pass_remove2])  # remove passenger from list



        ### BATTERY CALCULATION RATIO passenger distance over distance to passenger,
        # Strategy 4
        if strategy2 == 4:  # choosing score of passengers over battery

            prof_dist = []  # Calculate profit over distance to get there
            for i in range(len(battery_to_p)):
                if battery_ps_pe[i] != 0 and battery_to_p[i] != 0:
                    prof_dist.append(battery_ps_pe[i] / battery_to_p[i])
                else:  # if distance is 0 then profit is divided by 1
                    prof_dist.append(battery_ps_pe[i])

            loc2 = []
            # MINIMAX for BATTERY
            if minimax == 1:  # if minimax is turned on
                # if profit over distance to passenger multiplied by 1.5 is larger than the max profit possible
                if max(prof_dist) * 1.05 > max(battery_ps_pe):  # 1.05 is most optimal
                    pass_remove2 = prof_dist.index(max(prof_dist))  # choose the highest profit/distance passenger
                else:
                    pass_remove2 = battery_ps_pe.index(max(battery_ps_pe))
            else:
                pass_remove2 = prof_dist.index(max(prof_dist))  # Always choose this strategy, even if lower

            battery2 = battery_to_p[pass_remove2] + battery_ps_pe[pass_remove2]

            # loc2 = [passengers[battery_p.index(min(battery_p))][0][0],
            #                  passengers[battery_p.index(min(battery_p))][0][1]] #find passenger with lowest battery usage
            loc2 = passengers[pass_remove2][0]
            # pass_remove2 = battery_p.index(min(battery_p))  # index of chosen passenger (to be removed after picked up)
            score2 = 0
            # score2 = matrix[passengers[battery_p.index(min(battery_p))][0][0],
            #                  passengers[battery_p.index(min(battery_p))][0][1]]
            score2 = battery_ps_pe[pass_remove2]  # profit = distance of passenger start to end
            score_fin = score2
            battery_fin = battery2
            loc_fin = loc2
            passengers.remove(passengers[pass_remove2])  # remove passenger from list



        ### BATTERY CALCULATION, passenger with lowest battery demand
        # Strategy 5
        if strategy2 == 5:  # choosing score of passengers over battery

            #### ---- BEST ROUTE ----
            # Get all combination of passengers
            # We want to create a best-route-combination
            # However, only two passenger combinations will be explored, due to a lack of computational memory.
            # This method explores a passenger combination of the highest value and the third highest value (the second highest value will be chosen by the other player in the next turn)
            # First, we choose the highest value within the matrix, then calculate the move to the third highest value
            # Second, we choose the third highest value within the matrix, then calculate the move to the highest value
            # The best version with the least moves is selected.

            p_comb = []
            list1 = []
            list1 = battery_ps_pe.copy()  # used to create a simple increasing list without the highest value in the matrix
            list1.sort(reverse=True)

            if list1.count(list1[0]) > 3:  # if there are more than three equal highest values, algorithm could choose a third wrong location
                list1_idx = []  # getting indexes of highest values
                for i in range(len(battery_ps_pe)):
                    if battery_ps_pe[i] == list1[0]:
                        list1_idx.append(i)
                comb = list(permutations(list1_idx, 2))  # create combinations of two for all the highest values
            else:
                list1_idx = []  # getting indexes of highest values
                list1_idx2 = []  # getting indexes of highest values
                for i in range(len(battery_ps_pe)):
                    if battery_ps_pe[i] == list1[0]:
                        list1_idx.append(i)
                    if battery_ps_pe[i] == list1[2]:
                        list1_idx2.append(i)
                comb = list(permutations(list1_idx + list1_idx2, 2))  # otherwise just create combinations of the highest values and third highest value

            # for i in comb:
            #     p_comb.append(i)
            # formula to get the total length of the combinations
            best_route = []
            for i in comb:  # get first combination
                v = 0
                for j in range(len(i) - 1):  # Get distance of all end-start positions of passengers per combination

                    v += abs(passengers[i[j]][1][0] - passengers[i[j + 1]][0][
                        0])  # distance end and start passenger of comb in order
                    v += abs(passengers[i[j]][1][1] - passengers[i[j + 1]][0][
                        1])  # distance end and start passenger of comb in order

                v += abs(last_loc[0] - passengers[i[0]][0][0])  # distance from last location to first passenger in comb
                v += abs(last_loc[1] - passengers[i[0]][0][1])  # distance from last location to first passenger in comb

                for e in range(len(i)):  # Get distance per passenger
                    v += abs(passengers[e][0][0] - passengers[e][1][0])
                    v += abs(passengers[e][0][1] - passengers[e][1][1])

                # Append total distance (from start to passenger, between passengers and within) for each combination
                best_route.append(v)
            # ---- BEST ROUTE END-----

            # MINIMAX for BATTERY
            if minimax == 1:  # if minimax is turned on
                # if profit over distance to passenger multiplied by 1.5 is larger than the max profit possible
                if ((battery_ps_pe[comb[best_route.index(min(best_route))][0]] + battery_ps_pe[comb[best_route.index(min(best_route))][1]]) / 2) * 1.05 > max(battery_ps_pe):  # 1.05 is most optimal
                    pass_remove2 = comb[best_route.index(min(best_route))][0]  # choose the highest profit/distance passenger
                else:
                    pass_remove2 = battery_ps_pe.index(max(battery_ps_pe))
            else:
                pass_remove2 = comb[best_route.index(min(best_route))][0]  # Always choose this strategy, even if lower

            battery2 = battery_to_p[pass_remove2] + battery_ps_pe[pass_remove2]
            loc2 = passengers[pass_remove2][0]
            score2 = 0
            score2 = battery_ps_pe[pass_remove2]  # profit = distance of passenger start to end

            score_fin = score2
            battery_fin = battery2
            loc_fin = loc2
            passengers.remove(passengers[pass_remove2])  # remove passenger from list

        # Strategy 6
        if strategy2 == 6:  # always choosing highest score, then look at closest if more than one

            #### ---- BEST ROUTE ----
            # Get highest score
            # Find closest if more than one

            p_comb = []
            list1 = []
            list1 = battery_ps_pe.copy()  # used to create a simple increasing list without the highest value in the matrix
            list1.sort(reverse=True)

            # if there are more than three equal highest values, algorithm could choose a third wrong location
            list1_idx = []  # getting indexes of highest values
            for i in range(len(battery_ps_pe)):
                if battery_ps_pe[i] == list1[0]:
                    list1_idx.append(i)
            # if list1.count(list1[0]) > 1:
            # comb = list(permutations(list1_idx, len(list1_idx)))  # create combinations of two for all the highest values

            best_route = []   # calculates the distance for the highest profits
            for i in list1_idx:  # get first combination
                v = 0

                v += abs(last_loc[0] - passengers[i][0][0])  # distance from last location to first passenger in comb
                v += abs(last_loc[1] - passengers[i][0][1])  # distance from last location to first passenger in comb

                # Append total distance (from start to passenger, between passengers and within) for each combination
                best_route.append(v)
                # ---- BEST ROUTE END-----

            # MINIMAX for BATTERY
            if minimax == 1:  # if minimax is turned on
                # Choose the highest passenger value (profit) with the shortests distance, multiplied by 1.05 if larger than the max profit
                if ((battery_ps_pe[list1_idx[best_route.index(min(best_route))]] +
                     battery_ps_pe[list1_idx[best_route.index(min(best_route))]]) / 2) * 1.05 > max(battery_ps_pe):  # 1.05 is most optimal
                    pass_remove2 = list1_idx[best_route.index(min(best_route))]  # choose the highest profit/distance passenger
                else:
                    pass_remove2 = battery_ps_pe.index(max(battery_ps_pe))
            else:
                pass_remove2 = list1_idx[best_route.index(min(best_route))]  # Always choose this strategy, even if lower

            battery2 = battery_to_p[pass_remove2] + battery_ps_pe[pass_remove2]
            loc2 = passengers[pass_remove2][0]
            score2 = 0
            score2 = battery_ps_pe[pass_remove2]  # profit = distance of passenger start to end

            score_fin = score2
            battery_fin = battery2
            loc_fin = loc2
            passengers.remove(passengers[pass_remove2])  # remove passenger from list

    return score_fin, battery_fin, loc_fin, passengers


#-----------PLAY----------#
def play(count, ptot, size, last_loc1, last_loc2, player, mode, strategy1, strategy2, minimax):

    if count == 0:
        frames = []
        score_p1 = 0  # passenger score player 1, based on amount of passenger or density in cell
        score_p2 = 0  # passenger score player 2, based on amount of passenger or density in cell
        battery_p1 = 0  # battery usage score player 1, based on amount of moves (1 cell is one move)
        battery_p2 = 0  # battery usage score player 1, based on amount of moves (1 cell is one move)
        new_p = [[0,0]]  # using after first move to add passengers
        [matrix, density_matrix, passengers] = build_matrix(ptot, size)

    while sum(sum(matrix)) > 0:  # ends if there are no passengers left on matrix

        if player == 1:    # Player 1
            [matrix, density_matrix, battery_to_p, battery_ps_pe] = eridegame(matrix, passengers, last_loc1, new_p)   # calculate battery and update matrix
            [score_fin, battery_fin, loc_fin, passengers] = minimax_algorithm(matrix, passengers,  battery_to_p, battery_ps_pe, last_loc1, player, strategy1, strategy2, minimax)  # choose passenger
            score_p1 += score_fin   # update passsenger score
            battery_p1 += battery_fin  # update battery score
            matrix[loc_fin[0],loc_fin[1]] = matrix[loc_fin[0],loc_fin[1]]-1  # remove passenger

            [passengers,new_p] = add_passengers(passengers,size, mode)  # adding random passengers from 0-2 per move

            last_loc1 = loc_fin   # update to last location
            count += 1  # count moves
            player = 2    # switch player

        else:  # Player 2
            [matrix, density_matrix, battery_to_p, battery_ps_pe] = eridegame(matrix, passengers, last_loc2, new_p)   # calculate battery and update matrix
            [score_fin, battery_fin, loc_fin, passengers] = minimax_algorithm(matrix, passengers, battery_to_p, battery_ps_pe, last_loc2, player, strategy1, strategy2, minimax)   # choose passenger
            score_p2 += score_fin  # update passsenger score
            battery_p2 += battery_fin  # update battery score
            matrix[loc_fin[0],loc_fin[1]] = matrix[loc_fin[0],loc_fin[1]]-1  # remove passenger from matrix
            [passengers,new_p] = add_passengers(passengers,size, mode)  # adding random passengers from 0-2 per move
            last_loc2 = loc_fin  # update to last location
            count += 1    # count moves
            player = 1  # switch player

        ### VISUALISATION of passengers matrix
        # plt.imshow(matrix)
        # plt.colorbar()
        # plt.pause(0.3)
        # plt.clf()

        frames.append([matrix.copy(),last_loc1, score_p1, battery_p1, last_loc2, score_p2, battery_p2])   # for visualisation

        if count == ptot-10:  # end after ptot-10 moves to avoid errors if passengers run out fast
            matrix = np.zeros((size[1], size[0]))
        # switch to other player
        # player2(matrix,passengers, last_loc1,score_p1, battery_p1, last_loc2, score_p2, battery_p2)

    # return matrix, passengers, last_loc1, score_p1, battery_p1, last_loc2, score_p2, battery_p2
    return matrix, passengers, last_loc1, score_p1, battery_p1, last_loc2, score_p2, battery_p2, frames  # added frames for visualisation
# --------------END---------------------




### --------------SIMULATION---------------------
# SIMULATING GAME WITH FOCUS ON HIGH PASSENGER AMOUNTS
score_list1 = []   # score of passengers player 1
battery_list1 = []    # score of battery use player 1
score_list2 = []
battery_list2 = []
for r in range(0,sim):  # 100 simulations with scores appended in one list
    # insert: [counttime, total passengers, grid-size, starting location car1, starting location car2, player, passenger numbers change]
    [matrix, passengers, last_loc1, score_p1, battery_p1, last_loc2, score_p2, battery_p2, frames] = play(count, ptot, size, last_loc1,last_loc2, 1, change, strategy1, strategy2, minimax)
    score_list1.append(score_p1)  # score of passengers player 1
    battery_list1.append(battery_p1)   # score of battery use player 1
    score_list2.append(score_p2)    # score of passengers player 2
    battery_list2.append(battery_p2)   # score of battery use player 2
    last_loc1, last_loc2 = args.last_loc1, args.last_loc2   # reset starting location for next simulation


# Append all output into list: strategy_passenger
strategy_passenger = []
# strategy_passenger.append([score_list1,battery_list1,score_list2,battery_list2])   # use for multiple runs and combine in single file
strategy_passenger = [score_list1,battery_list1,score_list2,battery_list2]   # save output

if save_output:
    # # SAVE OUTPUT
    with open(str(ROOT)+"/strategy_passenger"+"_"+str(date), "wb") as fp:   #Pickling
        pickle.dump(strategy_passenger, fp)



# Print Statistics in terminal
print('FOCUS ON HIGH PASSENGER NUMBER')
print('player1 - Passenger score',statistics.mean(score_list1))
print('player1 - Battery score',statistics.mean(battery_list1))

print('FOCUS ON LOW BATTERY USAGE')
print('player1 - Passenger score',statistics.mean(score_list2))
print('player1 - Battery score',statistics.mean(battery_list2))

print('difference in Passenger score in %',"{:.2f}".format(
(statistics.mean(score_list1) - statistics.mean(score_list2) ) / statistics.mean(score_list1)*100 ))
print('difference in battery usage in %',"{:.2f}".format(
(statistics.mean(battery_list1) - statistics.mean(battery_list2) ) / statistics.mean(battery_list1)*100 ))
print('battery saving efficiency strategy player 2 over 1:',"{:.2f}".format(
    ((statistics.mean(battery_list1) - statistics.mean(battery_list2) ) / statistics.mean(battery_list1) ) / ((statistics.mean(score_list1) - statistics.mean(score_list2) ) / statistics.mean(score_list1) )
))

# Print all values in one row for easy copy paste into spreadsheet
# print([statistics.mean(score_list1),statistics.mean(battery_list1),
# statistics.mean(score_list2),statistics.mean(battery_list2)
# ])


