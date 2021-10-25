import sys

import numpy as np

from . import DATA_DIR


def get_gradients(
    size=["9m", "13m-city", "13m-coach", "13m-city-double", "13m-coach-double", "18m"]
):

    """Get gradient data as a Pandas `Series`.

    Gradients are given as km/h per second. Sourced from VECTO 3.3.7.

    :param name: The name of the driving cycle. "Urban delivery" is chosen by default if :param name: left unspecified.
    :type name: str

    ``name`` should be one of:

    * Urban delivery
    * Regional delivery
    * Long haul

    :returns: a_matrix pandas DataFrame object with driving time (in seconds) as index,
        and velocity (in km/h) as values.
    :rtype: panda.Series


    """

    dict_dc_sizes = {
        "9m": 1,
        "13m-city": 2,
        "13m-coach": 3,
        "13m-city-double": 4,
        "13m-coach-double": 5,
        "18m": 6,
    }

    try:
        list_col = [dict_dc_sizes[s] for s in size]
        arr = np.genfromtxt(DATA_DIR / "gradients.csv", delimiter=";")
        dc = arr[1:, list_col]
        dc = dc[~np.isnan(dc)]
        return dc.reshape((-1, len(list_col)))

    except KeyError:
        print("The specified driving cycle could not be found.")
        sys.exit(1)
