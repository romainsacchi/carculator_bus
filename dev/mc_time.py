import pandas as pd

from carculator_bus import *

bip = BusInputParameters()
# bip.stochastic(5)
bip.static()
dcts, array = fill_xarray_from_input_parameters(
    bip,
    scope={"year": [2020], "size": ["9m", "18m"], "powertrain": ["BEV-motion"]},
)
bm = BusModel(array, country="CH")
bm.set_all()
