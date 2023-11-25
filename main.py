"""
This file is the main file for the project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import json
import requests
from datetime import datetime
from datetime import timedelta
import random
import warnings

from src.forecast_number_car import process_data, plot_evolution, forecast_number_car, save_plot

warnings.filterwarnings("ignore")
## data on EV adoption from: https://www.bfs.admin.ch/bfs/en/home/statistics/mobility-transport/transport-infrastructure-vehicles/vehicles/road-vehicles-stock-level-motorisation.html

## Decide the consrtaint for the number of different vehicles present in 2050

data = pd.read_csv('data/ev_data.csv', encoding='ISO-8859-1')

data_ev = process_data(data)

data_ev_forecast = forecast_number_car(data_ev, year_constraint=2050, petrol_constraint=10000, diesel_constraint=0 , hybrid_constraint = 1e6, battery_elecrtic_constraint=5e6, pol_degree=3, weights = "exp").get_df()

# plot_evolution(data_ev_forecast)
save_plot(data_ev_forecast, "plots/evolution_number_car.png")