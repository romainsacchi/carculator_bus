import sys

import numpy as np

from . import DATA_DIR


def get_standard_driving_cycle(
    size=["9m", "13m-city", "13m-coach", "13m-city-double", "13m-coach-double", "18m"]
):

    """Get driving cycle data as a Pandas `Series`.

    Driving cycles are given as km/h per second. Sourced from VECTO 3.3.9.

    * "9m", "13m-city", "13-city-double", "18m": `Urban` driving cycle from VECTO
    * "13m-coach", "13m-coach-double": `Coach` driving cycle from VECTO

    :returns: a_matrix pandas DataFrame object with driving time (in seconds) as index,
        and velocity (in km/h) as values.
    :rtype: panda.Series

    """

    # definition of columns to select in the CSV file
    # each column corresponds to a size class
    # since the driving cycle is simulated for each size class
    # for example, a heavier truck will take more time to reach the target speed
    # because of higher inertia resistance

    dict_dc_sizes = {
        "9m": 1,
        "13m-city": 2,
        "13m-coach": 3,
        "13m-city-double": 4,
        "13m-coach-double": 5,
        "18m": 6,
        "Target speed, city": 7,
        "Target speed, coach": 8,
    }

    try:
        list_col = [dict_dc_sizes[s] for s in size]
        arr = np.genfromtxt(DATA_DIR / "driving_cycles.csv", delimiter=";")
        # we skip the headers
        dc = arr[1:, list_col]
        dc = dc[~np.isnan(dc)]
        return dc.reshape((-1, len(list_col)))

    except KeyError:
        print("The specified driving cycle could not be found.")
        sys.exit(1)
