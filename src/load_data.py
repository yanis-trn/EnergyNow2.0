"""
This file is used to load the data from the ACN dataset.
"""

## Necessaries libraries
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import json

def load_data() -> pd.DataFrame:
    """
    Load the data from the ACN dataset.
    """
    print("Loading data from the ACN dataset...")
    with open('data/acndata_sessions.json', 'r') as file:        data = json.load(file)

    items = data["_items"]

    # Create a list of dictionaries with the desired columns
    data_list = []
    for item in items:
        data_list.append({
            "connection_time": item["connectionTime"],
            "done_charging_time": item["doneChargingTime"],
            "kwh_delivered": item["kWhDelivered"]
        })

    print("number of charging events: {}".format(len(data["_items"])))
    # Create a DataFrame
    data_charging = pd.DataFrame(data_list)
    # Create a new column with just the time of the day

    data_charging["hour"] = pd.to_datetime(data_charging["connection_time"]).dt.strftime('%H:%M')

    # data_charging['15_min_interval'] = pd.to_datetime(data_charging['connection_time']).apply(lambda x: x - timedelta(minutes=x.minute % 15, seconds=x.second, microseconds=x.microsecond)).dt.strftime('%H:%M')

    data_charging['15_min_interval'] = pd.to_datetime(data_charging['connection_time']).apply(lambda x: (x + timedelta(hours=2) - timedelta(minutes=x.minute % 15, seconds=x.second, microseconds=x.microsecond)).strftime('%H:%M'))

    event_counts = data_charging.groupby('15_min_interval').size().reset_index(name='event_count')
    event_counts["probability"] = event_counts["event_count"] / event_counts["event_count"].sum()
    
    return data_charging, event_counts