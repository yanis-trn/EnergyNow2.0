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
from src.load_data import load_data
from src.utils import visualize_kwh_delivered, visualize_hourly_distribution, initialize_dataframe_cars, initialize_dataframe_states, get_number_EV, visualize_normal_charging, visualize_flex_charging
from src.model_charging import model_charging_normal, model_charging_flexibility

warnings.filterwarnings("ignore")

## This part is used to forecast the number of cars in the future.

# data on EV adoption from: https://www.bfs.admin.ch/bfs/en/home/statistics/mobility-transport/transport-infrastructure-vehicles/vehicles/road-vehicles-stock-level-motorisation.html
# Decide the consrtaint for the number of different vehicles present in 2050

data = pd.read_csv('data/ev_data.csv', encoding='ISO-8859-1')

data_ev = process_data(data)

data_ev_forecast = forecast_number_car(data_ev, year_constraint=2050, petrol_constraint=10000, diesel_constraint=0 , hybrid_constraint = 1e6, battery_elecrtic_constraint=5e6, pol_degree=3, weights = "exp").get_df()

# plot_evolution(data_ev_forecast)
# save_plot(data_ev_forecast, "plots/evolution_number_car.png")

##############################################################################################################################################################################
print("##############################################################################################################################################################################")

data_charging, event_counts = load_data()
visualize_kwh_delivered(data_charging)
visualize_hourly_distribution(event_counts)

year_simulation = int(input("Enter the year of simulation: "))

car_number_real = get_number_EV(data_ev_forecast, year_simulation)
car_number_simulated = 1000
ratio = car_number_real / car_number_simulated


print("Number of EV in {}: {}".format(year_simulation, car_number_real))

df_car = initialize_dataframe_cars(car_number_simulated)

df_simulation = initialize_dataframe_states(power=9, car_number=car_number_simulated, data_charging=data_charging, event_counts=event_counts)


df_summed, total_energy_needed = model_charging_normal(df_car, df_simulation, car_number_simulated, ratio)

visualize_normal_charging(df_summed, car_number_real, year_simulation, total_energy_needed, ratio)

print("##############################################################################################################################################################################")

max_power = int(input("Enter the maximum power consumption on the grid in Mwh: "))

df_summed_flex, total_energy_needed_flex = model_charging_flexibility(df_car, df_simulation, df_summed, car_number_simulated, max_power,time_regulation="00:00", quantity_regulation = 0, duration_regulation = 0, type_regulation = None, ratio = ratio)

visualize_flex_charging(df_summed_flex, car_number_real, year_simulation, total_energy_needed_flex, ratio)

time_regulation = str(input("Enter the time when regulation is needed: "))