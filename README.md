# EnergyNow2.0
This repository corresponds to the proposed solution from the SmartFlex team foor the EnergyNow 2.0 challenge. The challenge as been given by the Energy Science Center (ESC) from ETH and can be found [here](https://esc.ethz.ch/events/energy-now/challenges.html)

## Instalation 
Python 3.11.5 was used for this project
- conda env create -f environment.yaml
- conda activate EnergyNow

## Code structure

```
├── README.md
├── data
│   ├── acndata_sessions.json
│   └── ev_data.csv
├── plots
├── notebooks
│   ├── CKW-smartmeter-visualization-notebook.ipynb
│   └── EV-modeling.ipynb
├── src
│   ├── __init__.py
│   ├── forecast_number_car.py
│   ├── load_data.py
│   ├── model_charging.py
│   └── utils.py
└── main.py
```
## Demonstration video

![demonstration video](plots/DEMO%20EnergyNow.mov)

## Additionnal informations
Additionnal informations on the model can be found [here](https://drive.google.com/drive/folders/1VImr6ZG_6lgR7DtYY-NhQxtK-Vuy53o2?usp=share_link)
