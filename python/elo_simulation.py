import random
import joblib
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import dataframe_image as dfi
from elo_functions import overall_sim
from elo_functions import plot_dataframes

league_dict = {
    'Premier-League': '9',
    'La-Liga': '12',
    'Serie-A': '11',
    'Bundesliga': '20',
    'Ligue-1': '13'
}

#No. Simulations
n = 10000

#Max ELO gain/loss per match up
K = 32

simtabledict = overall_sim(league_dict, n, K)

plot_dataframes(simtabledict)