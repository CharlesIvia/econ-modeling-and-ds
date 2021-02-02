# Many data sources contain both numerical data and text
# Text can be used to create features for any prediction model but only
# ...after encoding text into some numerical reprentation


# AVALANChES

# Avalanches are a hazard in mountains and can be predicted based on
# ...snow conditions, weather and terrain

# WANT: Predict fatal accidents from the text of avalanche forecasts

# Since fatal accidents are rare, this prediction task will be quite difficult


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qeds
import json
import os
import urllib.request
import time
import requests
import io
import zipfile
import warnings

# activate plot theme
qeds.themes.mpl_style()


# Data

# Data from from Avalanche Canada API
# Rule of thumb: When considering scrapping a website check if the website has an API

# Incident data

url = "http://incidents.avalanche.ca/public/incidents/?format=json"

req = urllib.request.Request(url)

with urllib.request.urlopen(req) as response:
    result = json.loads(response.read().decode("utf-8"))

incident_list = result["results"]
# print(incident_list)

while result["next"] is not None:
    req = urllib.request.Request(result["next"])
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
    incident_list = incident_list + result["results"]

incident_brief = pd.DataFrame.from_dict(incident_list, orient="columns")
pd.options.display.max_rows = 20
pd.options.display.max_columns = 8

print(incident_brief)
