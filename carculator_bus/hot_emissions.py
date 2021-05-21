import numpy as np
import xarray
from . import DATA_DIR
import pickle


def _(o):
    """Add a trailing dimension to make input arrays broadcast correctly"""
    if isinstance(o, (np.ndarray, xarray.DataArray)):
        return np.expand_dims(o, -1)
    else:
        return o


def get_emission_factors():
    """ Emissions factors extracted for trucks from HBEFA 4.1
        deatiled by size, powertrain and EURO class for each substance.
    """
    fp = DATA_DIR / "hot_buses.pickle"

    with open(fp, 'rb') as f:
        hot = pickle.load(f)

    return hot


class HotEmissionsModel:
    """
    Calculate hot pollutants emissions based on HBEFA 4.1 data, function of speed (given by the driving cycle)
    for vehicles with a combustion engine.

    :param cycle: Driving cycle. Pandas Series of second-by-second speeds (km/h) or name (str)
        of cycle e.g., "Urban delivery", "Regional delivery", "Long haul".
    :type cycle: pandas.Series

    """

    def __init__(self, cycle):
        self.cycle = cycle
        self.em = get_emission_factors()

    def get_emissions_per_powertrain(
        self, powertrain_type, euro_classes, energy_consumption, size, debug_mode=False
    ):
        """
        Calculate hot pollutants emissions given a powertrain type (i.e., diesel, CNG) and a EURO pollution class,
        per air sub-compartment (i.e., urban, suburban and rural).

        The emission sums are further divided into `air compartments`: urban, suburban and rural.

        :param euro_classes:
        :param debug_mode:
        :param powertrain_type: "diesel", or "CNG"
        :type powertrain_type: str
        :param energy: second by second tank-to-wheel energy consumption
        :type energy: array
        :return: Pollutants emission per km driven, per air compartment.
        :rtype: numpy.array
        """

        if powertrain_type not in ("diesel", "cng"):
            raise TypeError("The powertrain type is not valid.")

        arr = self.em.sel(
            powertrain=powertrain_type,
            euro_class=euro_classes,
            component=[
                "HC",
                "CO",
                "NOx",
                "PM2.5",
                "CH4",
                "NMHC",
                "N2O",
                "NH3",
                "Benzene",
            ],
        ).transpose("component", "euro_class", "variable")

        distance = np.squeeze(self.cycle.sum(axis=0)) / 3600

        if isinstance(distance, np.float):
            distance = np.array(distance).reshape(1, 1)

        # Emissions for each second of the driving cycle equal:
        # a * energy consumption
        # with a being a coefficient  given by fitting HBEFA 4.1 data
        # the fitting of emissions function of energy consumption is described in the notebook
        # `HBEFA buses.ipynb` in the folder `dev`.
        a = arr.sel(variable="a").values[:, None, None, :, None, None] * energy_consumption.values

        # The receiving array should contain 40 substances, not 10
        arr_shape = list(a.shape)
        arr_shape[0] = 39
        em_arr = np.zeros(tuple(arr_shape))

        em_arr[:9] = a

        # Ethane, Propane, Butane, Pentane, Hexane, Cyclohexane, Heptane
        # Ethene, Propene, 1-Pentene, Toluene, m-Xylene, o-Xylene
        # Formaldehyde, Acetaldehyde, Benzaldehyde, Acetone
        # Methyl ethyl ketone, Acrolein, Styrene
        # which are calculated as fractions of NMVOC emissions

        ratios_NMHC = np.array([
            3.00E-04,
            1.00E-03,
            1.50E-03,
            6.00E-04,
            0.00E+00,
            0.00E+00,
            3.00E-03,
            0.00E+00,
            0.00E+00,
            0.00E+00,
            1.00E-04,
            9.80E-03,
            4.00E-03,
            8.40E-02,
            4.57E-02,
            1.37E-02,
            0.00E+00,
            0.00E+00,
            1.77E-02,
            5.60E-03
        ])


        em_arr[9:29] = em_arr[6]*ratios_NMHC[:, None, None, None, None, None]

        # remaining NMVOC
        em_arr[5] *= (1 - np.sum(ratios_NMHC))

        if powertrain_type == "diesel":
            # We also add heavy metals if diesel
            # which are initially defined per kg of fuel consumed
            # here converted to kg emitted/kj
            heavy_metals = np.array([
                1.83E-09,
                2.34E-12,
                2.34E-12,
                4.07E-08,
                4.95E-10,
                2.06E-10,
                7.01E-10,
                1.40E-12,
                1.24E-10,
                2.03E-10
            ])

            em_arr[29:] = heavy_metals.reshape(-1, 1, 1, 1, 1, 1) * energy_consumption.values

        # In case the fit produces negative numbers (it should not, though)
        em_arr[em_arr < 0] = 0

        # If the driving cycle selected is one of the driving cycles for which carculator has specifications,
        # we use the driving cycle "official" road section types to compartmentalize emissions.
        # If the driving cycle selected is instead specified by the user (passed directly as an array), we used
        # speed levels to compartmentalize emissions.

        urban = np.zeros((39, self.cycle.shape[-1], em_arr.shape[2], em_arr.shape[3], em_arr.shape[4]))
        suburban = np.zeros((39, self.cycle.shape[-1], em_arr.shape[2], em_arr.shape[3], em_arr.shape[4]))
        rural = np.zeros((39, self.cycle.shape[-1], em_arr.shape[2], em_arr.shape[3], em_arr.shape[4]))

        for s, x in enumerate(size):
            if x in ["9m", "13m-city", "13m-city-double"]:

                urban[:, s] = (np.sum(em_arr, axis=-1) / 1000 / distance[:, None, None, None])[:, s]

            else:

                suburban[:, s] = (np.sum(em_arr[..., 4000:12500], axis=-1) / 1000 / distance[:, None, None, None])[:, s]
                rural[:, s] = (np.sum(em_arr[..., 2000:4000], axis=-1) / 1000 / distance[:, None, None, None])[:, s]
                rural[:, s] += (np.sum(em_arr[..., 12500:], axis=-1) / 1000 / distance[:, None, None, None])[:, s]

        res = np.vstack((urban, suburban, rural))

        return res.transpose(1, 2, 0, 3, 4)