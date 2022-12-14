# E-Ride-Minimax


![presentation_image2](https://user-images.githubusercontent.com/84730858/192179820-1a25de27-6fe5-43b3-bbed-7a6ab225b200.jpg)

Keywords: AI in smart cities, climate change, decarbonizing, energy, game theory, Minimax, ride-sharing EV, smart economy, smart environment, smart mobility, traffic, transportation<br><br>

Paper publication: <b>Enriching Smart Cities by Optimizing Electric Vehicle Ride-Sharing through Game Theory</b><br> <br>

<div align="center">
Darko Radakovic<sup>1,</sup>, Anuradha Singh<sup>1,3</sup>, Aparna S. Varde<sup>1,2,3</sup>, and Pankaj Lal<sup>1,3</sup> <br>
1. Environmental Science and Management Ph.D. Program, Montclair State University, NJ, USA <br>
2. Department of Computer Science, Montclair State University, NJ, USA <br>
3. Clean Energy and Sustainability Analytics Center (CESAC), Montclair State University, NJ, USA  <br>
(radakovicd1  | singha2 | vardea | lalp)@montclair.edu <br>
ORCID ID: 0000-0002-3170-2510 (Varde) <br>
</div>
<br><br>

<p>
This MiniMax E-Ride sharing game allows two players to compete for passengers in a grid adapting the Minimax algorithm, treating EV ride-sharing companies as players. These players can use different strategies as described below. We hypothesize that one player choosing its next move via total passenger-travel distance (longer the distance, larger the profit); and another player via battery usage (ratio of total passenger-travel distance to vehicle-passenger distance: optimizing this ratio enables more travel without recharging.<br>
One can manipulate the number of passengers, the change in passenger numbers (decreasing, increasing or stable), the grid size (default is 5x10 cells), the player strategy mode, the player starting location, the number of simulations and whether to enable MiniMax. You can also select if you wish to save (as a pickle file) and/or plot the Scores and Battery usage (as JPEG).</p><br>

<h3>Google Colab script</h3>
<p><a href="https://colab.research.google.com/drive/1Pw6-R3JpJd3gdtlzSUDx0R_llnBCdY5W?usp=sharing" target="_blank">Ready to use Google Colab script</a></p><br><br>


<h3>Quick Start Examples</h3>

```bash
git clone https://github.com/darkoradakovic/E-Ride-Minimax
cd E-Ride-Minimax
pip install -r requirements.txt  # install
```

<h3>Inference</h3>
```bash
python e_ride_competition.py --sim 100 --save
                          --ptot  # number of passengers  (minimum ??50)
                          --size  # size of matrix  (x by y)
                          --change  # chose passengers numbers change: 'falling', 'stable', 'increasing'
                          --minimax  # decide if minimax algorithm is turned on for player2
                          --strategy1   #  choose from: '1' or '3'. (see below for overview)
                          --strategy2  # choose from: '2', '4', '5' or '6'
                          --last_loc1  # Player1 starting location
                          --last_loc2  # Player2 starting location
                          --sim  # Simulation total
                          --save  # save results to pickle file
```

<h3>STRATEGIES OVERVIEW</h3>
<ul>
<li>[Strategy 1] 'basic' [For player 1] find cell with most passengers, regardless their distance, chooses first passenger from list in this cell (old taxis without app strategy, by going to busy areas)</li>
<li>[Strategy 2] 'low' [For player 2] find closest passenger lowest distance, disregarding profit</li>
<li>[Strategy 3] 'passenger'[For player 1] find longest passenger start end (highest profit)</li>
<li>[Strategy 4] 'distance_ratio' [For player 2] choosing score of passengers over battery</li>
<li>[Strategy 5] 'battery' [For player 2] This method explores a passenger combination of the highest value and the third highest value (assuming that the second highest value will be chosen by the other player in the next turn), it decides whether to go first to the highest value or to the third highest values for the most optimal route.</li>
<li>[Strategy 6] 'passenger2' [For player 2] always choosing highest score, then look at closest if more than one  -> BEST RESULTS</li>
</ul>



<br><br><br>
<b>Copyright (c) 2022 Montclair State University</b>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

