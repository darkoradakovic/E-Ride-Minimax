# Authors: Darko Radakovic and Anuradha Singh
# 09/26/2022
# Montclair State University
# version 1 (Github ready)
# Copyright 2022, cite if using this code
# Part of e_ride_competition.py


import matplotlib.pyplot as plt
import datetime    # build in
import numpy as np

# used as suffix in file name when saving results, based local date and time
date = datetime.datetime.now().strftime("%x").replace('/','') + "_" + datetime.datetime.now().strftime("%X").replace(':','.')  # getting date for version

# Plotting boxplot Scores and Battery Usage
def boxplot(score_list1,score_list2,battery_list1,battery_list2):

    ### Boxplot scores
    fig = plt.figure(figsize=(10, 7))
    plt.title('Passenger Scores', size='18')
    plt.ylabel('Accumulated score based on amount of passengers\n on chosen location', size='12')
    # Creating plot
    plt.boxplot([score_list1, score_list2])
    plt.xticks([1, 2], ['Player 1\n Score', 'Player 2\n Score'])
    # show plot
    # plt.show()
    plt.savefig("minimax_score_results_"+str(date)+".jpg")

    # Boxplot batteries
    fig = plt.figure(figsize=(10, 7))
    plt.title('Battery usage', size='18')
    plt.ylabel(
        'Battery saved',
        size='12')
    # Creating plot
    plt.boxplot([battery_list1, battery_list2])
    plt.xticks([1, 2], ['Player 1\n Battery Saving', 'Player 2\n Battery Saving'])
    # show plot
    # plt.show()
    plt.savefig("minimax_battery_usage_results_"+str(date)+".jpg")



### VIDEO ANIMATION of passengers matrix
from matplotlib.ticker import AutoMinorLocator
import matplotlib.cm as cm  # VIDEO
import matplotlib.animation as animation  # VIDEO
import matplotlib.image as image

## Set plot-command to be drawn on specific commands such as plt.show()
# plt.ioff()   # set to False
# plt.ion()   # set to True
# print(plt.isinteractive())  # check if True or False
def animation(frames,size):
    fig = plt.figure(figsize=(15, 10))
    # fig = plt.figure(figsize=(10, 7),  dpi=1920/16)  # increase quality
    ax = fig.add_subplot()

    plt.title('Passengers per cell', size=20)
    plt.ylabel('latitude', size=16)
    plt.xlabel('longitude', size=16)
    plt.xticks(np.arange(0, size[0], 1))
    plt.yticks(np.arange(0, size[1], 1))
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    # Now hide the minor ticks (but leave the gridlines).
    ax.tick_params(which='minor', bottom=False, left=False)
    # Only show minor gridlines once in between major gridlines.
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # div = make_axes_locatable(ax)
    # cax = div.append_axes('right', '5%', '5%')
    cv0 = frames[0][0]
    im = ax.imshow(cv0, origin='lower')  # Here make an AxesImage rather than contour
    cb = fig.colorbar(im)

    # icon = image.imread('/content/car1.jpg')

    frames2 = []
    for i in range(len(frames)):  # i=3
        # frames2.append([plt.imshow(frames[i][0], animated=True)])  # Working
        if i <= 2:
            frames2.append([plt.imshow(frames[i][0], animated=True),
                            plt.text(frames[i][1][1], frames[i][1][0], 'CAR1', fontsize=14, fontweight='bold',
                                     color="white", ha='center', animated=True),
                            plt.text(frames[i][4][1], frames[i][4][0], 'CAR2', fontsize=14, fontweight='bold', color="red",
                                     ha='center', animated=True),
                            plt.text(0.2, 0.8, 'Round = {}'.format(i), fontsize=16, transform=plt.gcf().transFigure),
                            plt.text(0.2, 0.7, 'score pl1 = {}'.format(frames[i][2]), fontsize=16,
                                     transform=plt.gcf().transFigure),
                            plt.text(0.2, 0.65, 'score pl2 = {}'.format(frames[i][5]), fontsize=16,
                                     transform=plt.gcf().transFigure),
                            # plt.text(0.2, 0.6, 'score pl1 vs pl2 = {}%'.format( round(frames[i][2]/frames[i][3] *100,1) ), fontsize=16, transform=plt.gcf().transFigure),
                            plt.text(0.2, 0.5, 'battery use pl1 = {}'.format(frames[i][3]), fontsize=16,
                                     transform=plt.gcf().transFigure),
                            plt.text(0.2, 0.45, 'battery use pl2 = {}'.format(frames[i][6]), fontsize=16,
                                     transform=plt.gcf().transFigure),
                            # plt.text(0.2, 0.4, 'battery pl1 vs pl2= {}%'.format( round(frames[i][5]/frames[i][6] *100,1) ), fontsize=16, transform=plt.gcf().transFigure),
                            ])
        if i > 2:
            frames2.append([plt.imshow(frames[i][0], animated=True),
                            plt.text(frames[i][1][1], frames[i][1][0], 'CAR1', fontsize=14, fontweight='bold',
                                     color="white", ha='center', animated=True),
                            plt.text(frames[i][4][1], frames[i][4][0], 'CAR2', fontsize=14, fontweight='bold', color="red",
                                     ha='center', animated=True),
                            plt.text(0.2, 0.8, 'Round = {}'.format(i), fontsize=16, transform=plt.gcf().transFigure),
                            plt.text(0.2, 0.7, 'score pl1 = {}'.format(frames[i][2]), fontsize=16,
                                     transform=plt.gcf().transFigure),
                            plt.text(0.2, 0.65, 'score pl2 = {}'.format(frames[i][5]), fontsize=16,
                                     transform=plt.gcf().transFigure),
                            plt.text(0.2, 0.6, 'score pl2 vs pl1 = {}%'.format(round(frames[i][5] / frames[i][2] * 100, 1)),
                                     fontsize=16, transform=plt.gcf().transFigure),
                            plt.text(0.2, 0.5, 'battery use pl1 = {}'.format(frames[i][3]), fontsize=16,
                                     transform=plt.gcf().transFigure),
                            plt.text(0.2, 0.45, 'battery use pl2 = {}'.format(frames[i][6]), fontsize=16,
                                     transform=plt.gcf().transFigure),
                            plt.text(0.2, 0.4,
                                     'battery pl2 vs pl1= {}%'.format(round(frames[i][6] / frames[i][3] * 100, 1)),
                                     fontsize=16, transform=plt.gcf().transFigure),
                            ])
    # plt.show()

    ani = animation.ArtistAnimation(fig, frames2, interval=500, blit=True,
                                    repeat_delay=500)  # interval is speed, higher the slower
    #  blit parameter ensures that only those pieces of the plot are re-drawn which have been changed.
    # writervideo = animation.FFMpegWriter(fps=1, bitrate=5000)
    # ani.save('test_moviewritter.mp4', writer=writervideo)

    ani.save('test_movie.mp4')
    # plt.colorbar()
    plt.show()



import matplotlib as mpl
mpl.is_interactive()
