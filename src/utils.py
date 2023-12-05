"""
This file contains useful functions to be used in the project.
"""

## Necessaries libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime
from datetime import timedelta
import re

from src.model_charging import model_charging_constraint

def visualize_kwh_delivered(df):
    df = df.sort_values(by=['kwh_delivered'], ascending=False).reset_index(drop=True).copy()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df,x = df.index, y='kwh_delivered')
    plt.xticks(rotation=90)
    plt.xlabel('Number of sessions')
    plt.ylabel('kwh delivered')
    plt.title('Distribution of kwh delivered over sessions')
    plt.savefig('plots/distribution_kwh_delivered.png')
    # plt.show()
    plt.close()

def visualize_hourly_distribution(df):
    plt.figure(figsize=(15, 8))
    ax = plt.gca()  # Get the current axes

    sns.lineplot(data=df, x='15_min_interval', y='probability', ax=ax)
    ax.set_xlabel('Time of the day')
    ax.set_ylabel('Probability of charging session')
    ax.set_title('Probability distribution of charging sessions per 15 minutes interval')

    # Set the x-axis ticks and labels to show every Nth label
    N = 4  # Change N to adjust the frequency of labels
    labels = ax.get_xticks()[::N]
    ax.set_xticks(labels)
    ax.set_xticklabels(df['15_min_interval'][::N], rotation=45)
    plt.savefig('plots/probability_distribution_charging_sessions.png')
    # plt.show()
    plt.close()

def round_time_15(original_time: datetime) -> datetime:
    # Calculate the number of minutes to the next quarter hour
    minutes_to_next_quarter = (15 - (original_time.minute % 15)) % 15
    original_time = original_time.replace(second=0, microsecond=0)
    # Create a timedelta with the calculated minutes
    delta = timedelta(minutes=minutes_to_next_quarter)
    
    # Round the datetime to the next quarter hour
    rounded_dt = original_time + delta
    
    return rounded_dt

def sample_random_charging_event(data_charging, event_counts):
    sampled_time = np.random.choice(event_counts['15_min_interval'], p=event_counts['probability'])
    random_kwh = float(data_charging.sample(n=1)["kwh_delivered"])
    return datetime.strptime(sampled_time, '%H:%M'), random_kwh


def initialize_dataframe_states(power, car_number, data_charging, event_counts):
    '''
    Function to initialize the dataframe
    :input: power: float
    :input: car_number: int
    :output: df: DataFrame with the time and the power
    '''
    # Initialize DataFrame
    df_test = pd.DataFrame(columns=['car', 'plug_in_time', 'be_charged', 'pluged', 'energy_needed', 'power', 'last_time_to_charge'])


    # Loop over cars
    for car_number in range(1, car_number + 1):  # Assuming 5 cars for example
        # Sample charging event
        sampled_time, random_kwh = sample_random_charging_event(data_charging, event_counts)

        # Model charging constraint and add row to DataFrame
        df_test = pd.concat([df_test, pd.DataFrame([{
            'car': car_number,
            'plug_in_time': sampled_time.strftime("%H:%M"),
            'be_charged': round_time_15(model_charging_constraint(sampled_time, random_kwh, power)).strftime("%H:%M"),
            'pluged': False,
            'charging': False,
            'energy_needed': random_kwh,
            'power': power,
            'last_time_to_charge': (round_time_15(model_charging_constraint(sampled_time, random_kwh, power) - timedelta(hours=random_kwh / power))).strftime("%H:%M"),
            'time_before_need_to_charge': (round_time_15(model_charging_constraint(sampled_time, random_kwh, power) - timedelta(hours=random_kwh / power)) - sampled_time).seconds / 3600,
            'charge_done': False
        }])], ignore_index=True)

    # Display the resulting DataFrame order by time_before_need_to_charge
    df_test.sort_values(by=['time_before_need_to_charge'], inplace=True, ascending=True)
    df_test.reset_index(drop=True, inplace=True)
    df_test["car"] = df_test.index + 1

    return df_test


def initialize_dataframe_cars(car_number):

    start_time = pd.Timestamp("00:00")
    end_time = pd.Timestamp("23:59")
    date_range = pd.date_range(start_time, end_time, freq='15T').time

    # Convert time objects to string format "h:mm"
    time_strings = [time.strftime('%H:%M') for time in date_range]

    columns = ["car_" + str(i) for i in range(1, car_number + 1)] + ["total", "up", "down"]
    df = pd.DataFrame(columns=columns, index=time_strings)

    # Populate the DataFrame with 0 for each column
    for time in time_strings:
        row_data = {col: 0 for col in columns}
        
        # Append the row to the DataFrame
        df.loc[time] = row_data

    df = pd.concat([df, df], axis=0)

    return df

def get_number_EV(df: pd.DataFrame, year: int) -> int:
    return int(df[df["Year"] == str(year)]["battery_electric"].values[0])

def visualize_normal_charging(df_summed, car_number, year_simulation, total_energy_needed, ratio):
    plt.figure(figsize=(16, 6))

    columns_to_multiply = ["total"]
    df = df_summed.copy()
    df[columns_to_multiply] = df[columns_to_multiply].apply(lambda x: (x * ratio * 4) / 1000)
    # Plot "total", "up", and "down"
    df[['total']].plot(ax=plt.gca(), linestyle='--', marker='o')

    # Set labels and title
    plt.xlabel('hour of the day')
    plt.ylabel('Power in Mw')
    plt.title('Simulation of {} EV charging on the grid over a day in {}'.format(car_number, year_simulation))
    plt.text(0.05, 0.95, 'Total energy needed: {} Mwh'.format(round(total_energy_needed, 2)), transform=plt.gca().transAxes, ha='left', va='top', color='red', fontsize=12)

    # Show the plot
    plt.savefig('plots/normal_charging.png')
    plt.show()

def visualize_flex_charging(df_summed, car_number, year_simulation, total_energy_needed, ratio, path):
    plt.figure(figsize=(16, 6))

    df = df_summed.copy()
    columns_to_multiply = ["up","down","total"]
    df[columns_to_multiply] = df[columns_to_multiply].apply(lambda x: (x * ratio * 4) / 1000)
    # Plot "total", "up", and "down"
    df[['total', 'up', 'down']].plot(ax=plt.gca(), linestyle='--', marker='o')

    # Set labels and title
    plt.xlabel('hour of the day')
    plt.ylabel('Power in Mw')
    plt.title('Simulation of {} EV charging on the grid over a day in {}'.format(car_number, year_simulation))
    plt.text(0.05, 0.95, 'Total energy needed: {} Mwh'.format(round(total_energy_needed, 2)), transform=plt.gca().transAxes, ha='left', va='top', color='red', fontsize=12)


    # Show the plot
    plt.savefig(path)
    plt.show()

def generate_time_intervals(start_time, num_minutes):
    # Convert the start time to a datetime object
    start_datetime = datetime.strptime(start_time, '%H:%M')

    # Calculate the end time based on the number of minutes
    end_datetime = start_datetime + timedelta(minutes=num_minutes)

    # Generate 15-minute intervals
    interval = timedelta(minutes=15)
    current_time = start_datetime
    time_intervals = []

    while current_time < end_datetime:
        time_intervals.append(current_time.strftime('%H:%M'))
        current_time += interval

    return time_intervals


def get_flexibility_conditions():
    time_pattern = re.compile(r'^([01]\d|2[0-3]):([0-5]\d)$')
    while True:
        time_regulation = input("Enter the time when regulation is needed (HH:MM): ")
        
        if time_pattern.match(time_regulation):
            break
        else:
            print("Invalid format. Please enter the time in HH:MM format.")
    quantity_regulation = int(input("Enter the quantity of energy regulation needed in MW: "))
    duration_regulation = int(input("Enter the duration of the regulation in minutes: "))
    while True:
        type_regulation = input("Enter the type of regulation (up or down): ").lower()  # Convert input to lowercase for case-insensitivity
        
        if type_regulation in ["up", "down"]:
            break
        else:
            print("Invalid input. Please enter 'up' or 'down'.")
    
    return time_regulation, quantity_regulation, duration_regulation, type_regulation