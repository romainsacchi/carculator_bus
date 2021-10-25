import numpy as np

from .driving_cycles import get_standard_driving_cycle
from .gradients import get_gradients

np.seterr(divide="ignore", invalid="ignore")
import csv

import atmos
import xarray as xr

from . import DATA_DIR

MONTHLY_AVG_TEMP = "monthly_avg_temp.csv"


def _(o):
    """Add a trailing dimension to make input arrays broadcast correctly"""
    if isinstance(o, (np.ndarray, xarray.DataArray)):
        return np.expand_dims(o, -1)
    else:
        return o


def get_country_temperature(country):
    """
    Retrieves mothly average temperature
    :type country: country for which to retrieve temperature values
    :return:
    """

    with open(DATA_DIR / MONTHLY_AVG_TEMP, "r"):

        with open(DATA_DIR / MONTHLY_AVG_TEMP) as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                if row[2] == country:
                    return np.array([float(i) for i in row[3:]])

        print(
            f"Could not find monthly average temperature series for {country}. Uses those for CH instead."
        )

        with open(DATA_DIR / MONTHLY_AVG_TEMP) as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                if row[2] == "CH":
                    return np.array([int(i) for i in row[3:]])


def get_air_density(t):
    """
    Returns air density given the temperature
    :param t:
    :return:
    """
    return atmos.calculate("rho", Tv=t + 273.15, p=101325)


class EnergyConsumptionModel:
    """
    Calculate energy consumption of a vehicle for a given driving cycle and vehicle parameters.

    Based on a selected driving cycle, this class calculates the acceleration needed and provides
    two methods:

        - :func:`~energy_consumption.EnergyConsumptionModel.aux_energy_per_km` calculates the energy needed to power auxiliary services
        - :func:`~energy_consumption.EnergyConsumptionModel.motive_energy_per_km` calculates the energy needed to move the vehicle over 1 km

    Acceleration is calculated as the difference between velocity at t_2 and velocity at t_0, divided by 2.
    See for example: http://www.unece.org/fileadmin/DAM/trans/doc/2012/wp29grpe/WLTP-DHC-12-07e.xls

    :param cycle: Driving cycle. Pandas Series of second-by-second speeds (km/h) or name (str)
        of cycle e.g., "Urban delivery", "Regional delivery", "Long haul".
    :type cycle: pandas.Series
    :param rho_air: Mass per unit volume of air. Set to (1.225 kg/m3) by default.
    :type rho_air: float
    :param gradient: Road gradient per second of driving, in degrees. None by default. Should be passed as an array of
                    length equal to the length of the driving cycle.
    :type gradient: numpy.ndarray

    :ivar rho_air: Mass per unit volume of air. Value of 1.204 at 23C (test temperature for WLTC).
    :vartype rho_air: float
    :ivar velocity: Time series of speed values, in meters per second.
    :vartype velocity: numpy.ndarray
    :ivar acceleration: Time series of acceleration, calculated as increment in velocity per interval of 1 second,
        in meter per second^2.
    :vartype acceleration: numpy.ndarray


    """

    def __init__(
        self,
        cycle,
        size,
        country="CH",
        ambient_temp=None,
    ):
        # If a string is passed, the corresponding driving cycle is retrieved
        if isinstance(cycle, str):
            try:
                self.cycle_name = cycle
                cycle = get_standard_driving_cycle(size=size)

            except KeyError:
                raise KeyError("The driving cycle specified could not be found.")

        # if an array is passed instead, then it is used directly
        elif isinstance(cycle, np.ndarray):
            self.cycle_name = "custom"
            pass

        # if not, it's a problem
        else:
            raise TypeError("The format of the driving cycle is not valid.")

        self.gradient_name = self.cycle_name
        # retrieve road gradients (in degrees) for each second of the driving cycle selected
        self.gradient = get_gradients(size=size).reshape(-1, 1, 1, 1, len(size))
        # reshape the driving cycle
        self.cycle = cycle.reshape(-1, 1, 1, len(size))

        if ambient_temp is not None:
            self.t = np.resize(ambient_temp, (12,))
        else:
            self.t = get_country_temperature(country)

        self.rho_air = get_air_density(self.t).mean()

        # Unit conversion km/h to m/s
        self.velocity = (self.cycle * 1000) / 3600
        self.velocity = self.velocity[:, None, :, :, :]

        # Model acceleration as difference in velocity between time steps (1 second)
        # Zero at first value
        self.acceleration = np.zeros_like(self.velocity)
        self.acceleration[1:-1] = (self.velocity[2:] - self.velocity[:-2]) / 2

    def aux_energy_per_km(
        self,
        hvac_power,
        battery_cooling_unit,
        battery_heating_unit,
        indoor_temp=20,
    ):

        # use ambient temperature if provided, otherwise
        # monthly temperature average (12 values)

        # relation between ambient temperature
        # and HVAC power required
        # from https://doi.org/10.1016/j.energy.2018.12.064
        amb_temp = np.array([-30, -20, -10, 0, 10, 20, 30, 40])
        pct_power_HVAC = np.array([0.95, 0.54, 0.29, 0.13, 0.04, 0.08, 0.45, 0.7])

        # Heating power as long as ambient temperature, in W
        # is below the comfort indoor temperature
        p_heating = (
            np.where(
                self.t < indoor_temp, np.interp(self.t, amb_temp, pct_power_HVAC), 0
            ).mean()
            * hvac_power
        ).values

        # Cooling power as long as ambient temperature, in W
        # is above the comfort indoor temperature
        p_cooling = (
            np.where(
                self.t >= indoor_temp, np.interp(self.t, amb_temp, pct_power_HVAC), 0
            ).mean()
            * hvac_power
        ).values

        # We want to add power draw for battery cooling
        # and battery heating

        # battery cooling occurring above 20C, in W
        p_battery_cooling = np.where(
            self.t[:, None, None, None, None] > 20, battery_cooling_unit, 0
        ).mean(axis=-1)

        # battery heating occurring below 5C, in W
        p_battery_heating = np.where(
            self.t[:, None, None, None, None] < 5, battery_heating_unit, 0
        ).mean(axis=-1)
        return p_cooling, p_heating, p_battery_cooling, p_battery_heating

    def motive_energy_per_km(
        self,
        driving_mass,
        rr_coef,
        drag_coef,
        frontal_area,
        recuperation_efficiency=0,
        motor_power=0,
        debug_mode=False,
    ):
        """
        Calculate energy used and recuperated for a given vehicle per km driven.

        :param debug_mode:
        :param driving_mass: Mass of vehicle (kg)
        :type driving_mass: int
        :param rr_coef: Rolling resistance coefficient (dimensionless, between 0.0 and 1.0)
        :type rr_coef: float
        :param drag_coef: Aerodynamic drag coefficient (dimensionless, between 0.0 and 1.0)
        :type drag_coef: float
        :param frontal_area: Frontal area of vehicle (m2)
        :type frontal_area: float
        :param ttw_efficiency: Efficiency of translating potential energy into motion (dimensionless, between 0.0 and 1.0)
        :type ttw_efficiency: float
        :param recuperation_efficiency: Fraction of energy that can be recuperated (dimensionless, between 0.0 and 1.0). Optional.
        :type recuperation_efficiency: float
        :param motor_power: Electric motor power (watts). Optional.
        :type motor_power: int

        Power to overcome rolling resistance is calculated by:

        .. math::

            g v M C_{r}

        where :math:`g` is 9.81 (m/s2), :math:`v` is velocity (m/s), :math:`M` is mass (kg),
        and :math:`C_{r}` is the rolling resistance coefficient (dimensionless).

        Power to overcome air resistance is calculated by:

        .. math::

            \frac{1}{2} \rho_{air} v^{3} a_matrix C_{d}


        where :math:`\rho_{air}` is 1.225 (kg/m3), :math:`v` is velocity (m/s), :math:`a_matrix` is frontal area (m2), and :math:`C_{d}`
        is the aerodynamic drag coefficient (dimensionless).

        :returns: net motive energy (in kJ/km)
        :rtype: float

        """

        # Convert to km; velocity is m/s, times 1 second
        distance = self.velocity.sum(axis=0) / 1000

        ones = np.ones_like(self.velocity)

        # Resistance from the tire rolling: rolling resistance coefficient * driving mass * 9.81
        rolling_resistance = (driving_mass * rr_coef * 9.81).T.values * ones

        # Resistance from the drag: frontal area * drag coefficient * air density * 1/2 * velocity^2
        air_resistance = (
            frontal_area * drag_coef * self.rho_air / 2
        ).T.values * np.power(self.velocity, 2)

        # Resistance from road gradient: driving mass * 9.81 * sin(gradient)
        gradient_resistance = (driving_mass * 9.81).T.values * np.sin(self.gradient)

        # Inertia: driving mass * acceleration
        inertia = self.acceleration * driving_mass.values.T

        # Braking loss: when inertia is negative
        braking_loss = np.where(inertia < 0, inertia * -1, 0)

        total_resistance = (
            rolling_resistance + air_resistance + gradient_resistance + inertia
        )

        if not debug_mode:

            # Power required: total resistance * velocity
            total_power = total_resistance * self.velocity / 1000

            # Recuperation of the braking power within the limit of the electric engine power
            recuperated_power = braking_loss * self.velocity / 1000

            return (
                total_power.astype("float32"),
                recuperated_power.astype("float32"),
                distance,
            )

        # if `debug_mode` == True, returns instead
        # the power to overcome rolling resistance, air resistance, gradient resistance,
        # inertia and braking resistance, as well as the total power and the energy to overcome it.
        else:
            rolling_resistance *= self.velocity / 1000
            air_resistance *= self.velocity / 1000
            gradient_resistance *= self.velocity / 1000
            inertia *= self.velocity / 1000
            braking_loss *= self.velocity / 1000
            total_resistance = (
                rolling_resistance
                + air_resistance
                + gradient_resistance
                + inertia
                + braking_loss
            )

            recuperated_power = braking_loss * recuperation_efficiency.values.T
            recuperated_power = np.clip(recuperated_power, 0, motor_power.values.T)

            return (
                xr.DataArray(
                    np.squeeze(rolling_resistance),
                    dims=["values", "year", "powertrain", "size"],
                ),
                xr.DataArray(
                    np.squeeze(air_resistance),
                    dims=["values", "year", "powertrain", "size"],
                ),
                xr.DataArray(
                    np.squeeze(gradient_resistance),
                    dims=["values", "year", "powertrain", "size"],
                ),
                xr.DataArray(
                    np.squeeze(inertia), dims=["values", "year", "powertrain", "size"]
                ),
                xr.DataArray(
                    np.squeeze(braking_loss),
                    dims=["values", "year", "powertrain", "size"],
                ),
                xr.DataArray(
                    np.squeeze(recuperated_power),
                    dims=["values", "year", "powertrain", "size"],
                ),
                xr.DataArray(
                    np.squeeze(total_resistance),
                    dims=["values", "year", "powertrain", "size"],
                ),
            )
