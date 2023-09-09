import pandas as pd
import numpy as np
import joblib
from urllib.request import urlopen

print("load model")
model = joblib.load("static/GridSearchCV.pkl")
print("model loaded")

def predictor(data):
    #model = joblib.load("GridSearchCV.pkl")
    cols=["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
    sample = pd.DataFrame([data], columns=cols)

    sample["CDP"] = sample["CDP"] * 1000  # Conversion to mbar
    sample["AT"] = sample["AT"] + 273  # Conversion to Kelvin
    sample["TIT"] = sample["TIT"] + 273  # Conversion to Kelvin
    sample["TAT"] = sample["TAT"] + 273  # Conversion to Kelvin

    gamma_air = 1.4  # Usual value to assume for specific heats ratio of air
    sample["TAC"] = np.power(sample["CDP"] / sample["AP"], (gamma_air - 1) / gamma_air) * sample["AT"]

    ## Brayton Cycle Efficeincy via Temperature Changes
    # Ref: https://web.mit.edu/16.unified/www/SPRING/propulsion/notes/node27.html

    sample["IBCE"] = 1 - sample["AT"] / sample["TAC"]  # Ideal Brayton Cycle Efficiency
    sample["RBCE"] = 1 - (sample["TAT"] - 1 / sample["AT"]) / (sample["TIT"] - 1 / sample["TAC"])  # Real Brayton Cycle Efficiency
    sample["ER"] = sample["RBCE"] / sample["IBCE"]  # Real operation efficiency ratio

    ## Square root of CDP
    # Ref: https://www.ge.com/content/dam/gepower-new/global/en_US/downloads/gas-new-site/resources/reference/ger-4211-gas-turbine-emissions-and-control.pdf

    sample["SRCDP"] = np.sqrt(sample["CDP"])  # Square root of CDP
    return model.predict(sample)