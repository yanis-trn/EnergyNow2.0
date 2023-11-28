"""
This file contains the model fot the charging sessions.
"""

## Necessaries libraries
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


def model_charging(start_charging_time, required_energy, power_charge):
    from src.utils import round_time_15
    '''
    Situation with no flexibility
    This function model the charging of an EV with a given power and a given required energy when just charging when plugged in
    :input: start_charging_time: datetime object
    :input: required_energy: float
    :input: power_charge: float
    :output: df: DataFrame with the time and the power'''
    start_time = pd.Timestamp("00:00")
    end_time = pd.Timestamp("23:59")
    time_intervals = pd.date_range(start_time, end_time, freq='15T').time

    # Create a DataFrame with 'time' and 'power' columns
    df = pd.DataFrame({'time': time_intervals, 'power': 0})

    round_start_time = round_time_15(start_charging_time)
    
    # Calculate the number of 15-minute intervals required to charge the car
    intervals = int(np.ceil((required_energy*4) / power_charge))
    # print("number of intervals: {}".format(intervals))

    # Calculate the end time
    end_charging_time = round_start_time + timedelta(minutes=15*intervals)

    # print("time of end charging: {}".format(end_charging_time))

    # deals with the case when the charging as to end the next day
    if end_charging_time.date() > round_start_time.date():
        df.loc[df['time'] < end_charging_time.time(), 'power'] = power_charge
        df.loc[df['time'] >= round_start_time.time(), 'power'] = power_charge

    else:
        df.loc[(df['time'] >= round_start_time.time()) & (df['time'] < end_charging_time.time()), 'power'] = power_charge

    return df


def model_charging_constraint(start_charging_time, required_energy, power_charge):
    '''
    Function to model the constraint for the charging habits of the swiss population
    :input: start_charging_time: datetime object
    :return: end_charging_constraint: datetime object
    '''
    # If the charging start after 16h00 we can hipothetize that the car has to be charged latest at 7h00 (case of people charging at home)
    # If the charging start after 7h00 we can hipothetize that the car has to be charged latest at 16h00 (case of people charging at work)
    # If the charging can not finish before 7h00 or 16h00 then we hipothesizee that the car has to be charged in 3 hours. If it does not have the time then it has to charge as fast as possible

    delta_hours = 3
    if start_charging_time.hour >= 16 or start_charging_time.hour < 4:
        end_charging_constraint = datetime.strptime("7:00", "%H:%M")
        delta = (datetime.strptime("7:00", "%H:%M") - start_charging_time).seconds/3600
        if required_energy >= power_charge*delta:
            end_charging_constraint = start_charging_time + timedelta(hours=required_energy/power_charge)
    elif start_charging_time.hour >= 7 and start_charging_time.hour < 15:
        end_charging_constraint = datetime.strptime("16:00", "%H:%M")
        delta = (datetime.strptime("16:00", "%H:%M") - start_charging_time).seconds/3600
        if required_energy >= power_charge*delta_hours or required_energy >= power_charge*delta:
            end_charging_constraint = start_charging_time + timedelta(hours=required_energy/power_charge)
    else:
        if required_energy >= power_charge*delta_hours:
            end_charging_constraint = start_charging_time + timedelta(hours=required_energy/power_charge)
        else:
            end_charging_constraint = start_charging_time + timedelta(hours=delta_hours)
        

    # Format the datetime object as a string in "%H:%M" format
    end_charging_constraint_str = end_charging_constraint.strftime("%H:%M")
    
    return end_charging_constraint
        

def model_charging_normal(df_car,df_simulation, car_number, ratio):
    """
    This function model the charging of an EV with a given power and a given required energy when just charging when plugged in
    :input: df_car: DataFrame with the cars
    :input: df_simulation: DataFrame with the simulation
    :input: car_number: int
    :output: df: DataFrame with the time and the power
    """
    df_prov = df_simulation.copy()
    df = df_car.copy()
    car_rows = {car: df_prov.loc[df_prov['car'] == car] for car in range(1, car_number+1)}

    for time_interval, row_df in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        down_regulation = 0

        for car in range(1, car_number+1):
            row = car_rows[car]

            # Situation when the car is pluged and can start to be charged
            if row["plug_in_time"].item() == time_interval and not row["charge_done"].item():
                row["pluged"] = True
                row["charging"] = True

            # Situation when the car do not need to be charged anymore => charged done
            if row["energy_needed"].item() <= 0 and not row["charge_done"].item():
                row["pluged"] = False
                row["charging"] = False
                row["charge_done"] = True

            # Situation when the car is pluged and start to charge
            if row["pluged"].item():
                row["charging"] = True

                if row["charging"].item():
                    row["last_time_to_charge"] = (datetime.strptime(row["last_time_to_charge"].item(), "%H:%M") + timedelta(minutes=15)).strftime("%H:%M")
                    row_df["car_" + str(car)] = min(row["power"].item()/4, row["energy_needed"].item())

                    if row["time_before_need_to_charge"].item() > 0:
                        down_regulation += min(row["power"].item()/4, row["energy_needed"].item())

                    row["energy_needed"] = max(row["energy_needed"].item() - row["power"].item()/4, 0)

            if row["pluged"].item() and not row["charge_done"].item() and not row["charging"].item():
                row["time_before_need_to_charge"] = max(row["time_before_need_to_charge"].item() - 0.25, 0)

    row_df["down"] = down_regulation
    df_summed = df.groupby(df.index).sum()
    df_summed['total'] = df_summed.filter(like='car_').sum(axis=1)
    df_summed_modif = df_summed.copy()
    columns_to_multiply = ["down", "total"]
    df_summed_modif[columns_to_multiply] = df_summed_modif[columns_to_multiply].apply(lambda x: (x * ratio) / 1000)
    total_energy_needed = df_summed_modif['total'].sum()
    

    return df_summed, total_energy_needed


def model_charging_flexibility(df_car, df_simulation, df_normal, car_number, max_pow,time_regulation, quantity_regulation, duration_regulation, type_regulation, ratio):
    from src.utils import generate_time_intervals
    """
    This function model the charging of an EV with a given power and a given required energy when just charging when plugged in
    :input: df_car: DataFrame with the cars
    :input: df_simulation: DataFrame with the simulation
    :input: df_summed: DataFrame with the summed cars
    :input: car_number: int
    :input: max_power: int
    :input: time_regulation: datetime object
    :input: quantity_regulation: int
    :input: duration_regulation: int
    :input: type_regulation: str
    :input: ratio: float
    :output: df: DataFrame with the time and the power
    """
    list_regulation = generate_time_intervals(time_regulation, duration_regulation)

    df_prov = df_simulation.copy()
    df = df_car.copy()
    car_rows = {car: df_prov.loc[df_prov['car'] == car] for car in range(1, car_number+1)}
    # max_power = max_pow * (1000 / (ratio * 4))
    quantity_regulation = quantity_regulation * (1000 / (ratio * 4))
    # print("max_power: {}".format(max_power))

    for time_interval, row_df in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        down_regulation = 0
        up_regulation = 0
        power_time_interval = 0
        # print("time_interval: {}".format(time_interval))

        if time_interval in list_regulation:
            # print("time_interval: {}".format(time_interval))
            total_power = df_normal.loc[time_interval]["total"]
            # print("total_power: {}".format(total_power))
            if type_regulation == "down":
                max_power = total_power - quantity_regulation
                # print("max_power: {}".format(max_power))
            elif type_regulation == "up":
                max_power = total_power + quantity_regulation
                # print("max_power: {}".format(max_power))
        else:
            max_power = max_pow * (1000 / (ratio * 4))

        for car in range(1, car_number+1):
            row = car_rows[car]

            # Situation when the car is pluged and can start to be charged
            if row["plug_in_time"].item() == time_interval and not row["charge_done"].item():
                row["pluged"] = True
                row["charging"] = True

            # Situation when the car do not need to be charged anymore => charged done
            if row["energy_needed"].item() <= 0 and not row["charge_done"].item():
                row["pluged"] = False
                row["charging"] = False
                row["charge_done"] = True

            # Situation when power to high so down regulation
            if power_time_interval >= max_power and row["pluged"].item():
                if row["time_before_need_to_charge"].item() == 0:
                    raise Exception("time before need to charge is 0 for car number {}".format(car))
                row["charging"] = False
                up_regulation += min(row["power"].item()/4, row["energy_needed"].item())
                row_df["car_" + str(car)] = 0

            # Situation when the car is pluged and start to charge
            if row["pluged"].item() and power_time_interval < max_power:
                row["charging"] = True

                if row["charging"].item():
                    row["last_time_to_charge"] = (datetime.strptime(row["last_time_to_charge"].item(), "%H:%M") + timedelta(minutes=15)).strftime("%H:%M")
                    row_df["car_" + str(car)] = min(row["power"].item()/4, row["energy_needed"].item())
                    power_time_interval += row_df["car_" + str(car)]

                    if row["time_before_need_to_charge"].item() > 0:
                        down_regulation += min(row["power"].item()/4, row["energy_needed"].item())

                    row["energy_needed"] = max(row["energy_needed"].item() - row["power"].item()/4, 0)

            if row["pluged"].item() and not row["charge_done"].item() and not row["charging"].item():
                row["time_before_need_to_charge"] = max(row["time_before_need_to_charge"].item() - 0.25, 0)
        row_df["down"] = down_regulation
        row_df["up"] = up_regulation

    df_summed = df.groupby(df.index).sum()
    df_summed['total'] = df_summed.filter(like='car_').sum(axis=1)
    df_summed["down"] = df_summed["total"] - df_summed["down"]
    df_summed["down"] = df_summed["down"].clip(lower=0)
    df_summed["up"] = df_summed["up"] + df_summed["total"]


    df_summed_modif = df_summed.copy()
    columns_to_multiply = ["up", "down", "total"]
    df_summed_modif[columns_to_multiply] = df_summed_modif[columns_to_multiply].apply(lambda x: (x * ratio) / 1000)
    total_energy_needed = df_summed_modif['total'].sum()
    print("total energy needed over the day: {} Mwh".format(round(total_energy_needed,2)))

    return df_summed, total_energy_needed