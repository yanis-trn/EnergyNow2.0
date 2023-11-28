import PySimpleGUI as sg
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import warnings
import re
import seaborn as sns
import io
import sys

from src.forecast_number_car import process_data, plot_evolution, forecast_number_car, save_plot
from src.load_data import load_data
from src.utils import visualize_kwh_delivered, visualize_hourly_distribution, initialize_dataframe_cars, initialize_dataframe_states, get_number_EV, visualize_normal_charging, visualize_flex_charging, get_flexibility_conditions
from src.model_charging import model_charging_normal, model_charging_flexibility

warnings.filterwarnings("ignore")
plt.ioff()


def calculate(data):
    return data['Value'] * 2

def redirect_stdout_to_multiline(output_element):
    class StdoutRedirector(io.TextIOBase):
        def write(self, string):
            output_element.update(value=f"{output_element.get()}{string}")

    sys.stdout = StdoutRedirector()

def main():
    # Define the layout of the GUI
    layout = [
        [sg.Button("Run Script")],
        [sg.Output(size=(100, 10), key="output_label")],
        [sg.Canvas(key="plot_canvas1"), sg.Canvas(key="plot_canvas2"), sg.Canvas(key="plot_canvas3")]
    ]

    # Create the window
    window = sg.Window("Data Analyzer", layout, resizable=True)
    output_text_elem = window["output_label"]  # Corrected key

    # Redirect stdout to the multiline element
    redirect_stdout_to_multiline(output_text_elem)

    data = None
    result = None

    # Event loop
    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        elif event == "Run Script":
            data = pd.read_csv('data/ev_data.csv', encoding='ISO-8859-1')
            data_ev = process_data(data)
            output_text_elem.update("Loading historical data on EV adoption...\n")
            output_text_elem.update("Forecasting the number of EV in Switzerland up to 2050...\n")
            data_ev_forecast = forecast_number_car(data_ev, year_constraint=2050, petrol_constraint=10000, diesel_constraint=0 , hybrid_constraint = 1e6, battery_elecrtic_constraint=5e6, pol_degree=3, weights = "exp").get_df()

            plt.figure()
    
            sns.lineplot(data=data_ev_forecast, x='Year', y='Petrol', label='Petrol')
            sns.lineplot(data=data_ev_forecast, x='Year', y='Diesel', label='Diesel')
            sns.lineplot(data=data_ev_forecast, x='Year', y='hybrid', label='Hybrid')
            sns.lineplot(data=data_ev_forecast, x='Year', y='battery_electric', label='Battery electric')
            sns.lineplot(data=data_ev_forecast, x='Year', y='total_car', label='Total')
            
            # Set the x-ticks and labels for every 2 years
            x_ticks = data_ev_forecast['Year'].unique()[::2]
            x_tick_labels = [str(year) for year in x_ticks]
            
            plt.xticks(ticks=x_ticks, labels=x_tick_labels)
            plt.xticks(ticks=x_ticks, labels=x_tick_labels, rotation=45)

            plt.legend()
            plt.xlabel('Year')
            plt.ylabel('Number of cars')
            canvas_elem = window["plot_canvas1"]
            draw_figure(canvas_elem, plt.gcf())

            save_plot(data_ev_forecast, "plots/evolution_number_car.png")

            ###############################################

            data_charging, event_counts = load_data()
            # visualize_kwh_delivered(data_charging)
            # visualize_hourly_distribution(event_counts)

            year_simulation = sg.popup_get_text('Enter the year of simulation:', title='Year Input', default_text='')
            output_text_elem.update(f"Simulation Year: {year_simulation}\n")

            ratio_car_charging = 0.8
            car_number_real = get_number_EV(data_ev_forecast, year_simulation)*ratio_car_charging
            car_number_simulated = 1000
            ratio = car_number_real / car_number_simulated
            output_text_elem.update("Number of EV in {}: {}\n".format(year_simulation, car_number_real))
            print("Number of EV in {}: {}".format(year_simulation, car_number_real))
        
            df_car = initialize_dataframe_cars(car_number_simulated)
            df_simulation = initialize_dataframe_states(power=9, car_number=car_number_simulated, data_charging=data_charging, event_counts=event_counts)

            df_summed, total_energy_needed = model_charging_normal(df_car, df_simulation, car_number_simulated, ratio)
            output_text_elem.update("total energy needed over the day: {} Mwh\n".format(round(total_energy_needed,2)))
            # visualize_normal_charging(df_summed, car_number_real, year_simulation, total_energy_needed, ratio)


    window.close()

def draw_figure(canvas_elem, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, master=canvas_elem.Widget)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

if __name__ == "__main__":
    main()