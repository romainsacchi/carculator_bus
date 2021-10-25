import numexpr as ne
import numpy as np
import xarray as xr
from prettytable import PrettyTable

from .background_systems import BackgroundSystemModel
from .driving_cycles import get_standard_driving_cycle
from .energy_consumption import EnergyConsumptionModel
from .hot_emissions import HotEmissionsModel
from .noise_emissions import NoiseEmissionsModel
from .particulates_emissions import ParticulatesEmissionsModel


def finite(array, mask_value=0):
    return np.where(np.isfinite(array), array, mask_value)


class BusModel:

    """
    This class represents the entirety of the vehicles considered, with useful attributes, such as an array that stores
    all the vehicles parameters.

    :ivar array: multi-dimensional numpy-like array that contains parameters' value(s)
    :vartype array: xarray.DataArray
    :ivar country: ISO-2 country code
    :vartype country: str
    :ivar energy_target: energy reduction target to achieve, relative to 2020. Should be of format {2025: 0.85, 2030: 0.7}.
    :ivartype energy_target: dict
    :ivar ambient_temp: ambient temperature to use to calculate heating and cooling power demand
    :ivartype ambient_temp: int
    """

    def __init__(
        self,
        array,
        country=None,
        fuel_blend=None,
        energy_target={
            2025: 0.85,
            2030: 0.7,
        },  # <-- these are energy targets to achieve for ICE vehicles
        energy_storage=None,
        ambient_temp=None,
    ):

        self.array = array
        self.bs = BackgroundSystemModel()

        self.country = country or "CH"  # <-- defaults to Switzerland if not specified
        self.fuel_blend = self.define_fuel_blends(fuel_blend)
        self.energy_storage = energy_storage or {
            "electric": {
                "BEV-opp": "LTO",
                "BEV-depot": "NMC-111",
                "BEV-motion": "LTO",
                "FCEV": "NMC-111",
                "HEV-d": "NMC-111",
            }
        }

        self.cycle = get_standard_driving_cycle(size=array.coords["size"].values)

        self.energy_target = energy_target

        self.energy = None

        self.ecm = EnergyConsumptionModel(
            cycle=self.cycle,
            size=self.array.coords["size"].values,
            ambient_temp=ambient_temp,
            country=self.country,
        )

        if ambient_temp is not None:
            if isinstance(ambient_temp, int):
                self.ambient_temp = ambient_temp
            elif isinstance(ambient_temp, float):
                self.ambient_temp = int(ambient_temp)
            elif isinstance(ambient_temp, str):
                try:
                    self.ambient_temp = int(ambient_temp)
                except ValueError:
                    raise ValueError(
                        "The ambient temperature value `ambient_temp` should be an integer value, "
                        f"not {type(ambient_temp)}."
                    )
            else:
                raise ValueError(
                    "The ambient temperature value `ambient_temp` should be an integer value, "
                    f"not {type(ambient_temp)}."
                )
        else:
            self.ambient_temp = None

    def __call__(self, key):
        """
        This method fixes a dimension of the `array` attribute given a powertrain technology selected.

        Set up this class as a context manager, so we can have some nice syntax

        .. code-block:: python

            with class('some powertrain') as cpm:
                cpm['something']. # Will be filtered for the correct powertrain

        On with block exit, this filter is cleared
        https://stackoverflow.com/a/10252925/164864

        :param key: a_matrix powertrain type, e.g., "FCEV"
        :type key: str
        :return: An instance of `array` filtered after the powertrain selected.

        """
        self.__cache = self.array
        self.array = self.array.sel(powertrain=key)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.array = self.__cache
        del self.__cache

    def __getitem__(self, key):
        """
        Make class['foo'] automatically filter for the parameter 'foo'
        Makes the model code much cleaner

        :param key: Parameter name
        :type key: str
        :return: `array` filtered after the parameter selected
        """

        return self.array.sel(parameter=key)

    def __setitem__(self, key, value):
        self.array.loc[{"parameter": key}] = value

    def set_all(self):
        """
        This method runs a series of other methods to obtain the tank-to-wheel energy requirement, efficiency
        of the car, costs, etc.

        :meth:`set_component_masses()`, :meth:`set_car_masses()` and :meth:`set_power_parameters()` and
         :meth:`set_energy_stored_properties` relate to one another.
        `powertrain_mass` depends on `power`, `curb_mass` is affected by changes in `powertrain_mass`,
        `combustion engine mass`, `electric engine mass`. `energy battery mass` is influenced by the `curb mass` but also
        by the `daily distance` the truck has. `power` is also varying with `curb_mass`.

        The current solution is to loop through the methods until either:
        * the change in curb mass between two iterations is inferior to 2%.
        * or the number of vehicles with a solution has been stable for the last 3 iterations
        * and that there has been at least 7 iterations
        It is then assumed that the buses are correctly sized.

        :returns: Does not return anything. Modifies ``self.array`` in place.

        """

        diff = 1
        arr = np.array([])

        # whether the vehicles are compliant
        self["is_compliant"] = True
        # whether the technology is available for the given year
        self["is_available"] = True

        print("Finding solutions for buses...")

        while abs(diff) > 0.02 or np.std(arr[-3:]) > 0.05 and len(arr) < 7:

            # driving mass from the previous iteration
            old_driving_mass = self["driving mass"].sum().values

            # calculate the curb and driving mass of the vehicles
            self.set_car_masses()
            # size the engines and motors
            self.set_power_parameters()
            # sizes different vehicle components
            self.set_component_masses()
            # adjust parameters for the fuel cell stack
            self.set_fuel_cell_parameters()
            # calculate the TtW energy
            self.calculate_ttw_energy()
            # sizes the batteries
            # and fuel tanks
            # given the newly calculated TtW energy
            self.set_battery_fuel_cell_replacements()
            self.set_energy_stored_properties()
            self.set_battery_size()

            if self.energy_target is not None:
                if len(self.array.year.values) > 1:
                    # if there are vehicles after 2020, we need to ensure CO2 standards compliance
                    # return an array with non-compliant vehicles
                    non_compliant_vehicles = self.adjust_combustion_power_share()
                    arr = np.append(arr, non_compliant_vehicles.sum())
                else:
                    arr = np.append(arr, [0])
                    non_compliant_vehicles = 0
            else:
                arr = np.append(arr, [0])
                non_compliant_vehicles = 0

            diff = (self["driving mass"].sum().values - old_driving_mass) / self[
                "driving mass"
            ].sum()

        self.adjust_cost()
        self.set_electricity_consumption()
        self.set_costs()
        # defines hot pollutant emissions
        # along the driving cycle
        self.set_hot_emissions()
        # defines abrasion emissions
        self.set_particulates_emission()
        # defines noise emissions
        self.set_noise_emissions()
        self.check_compliance_of_buses()

        print("")
        print("Number of passengers on board")
        print(
            "'X' BEV with driving mass when fully occupied superior to the permissible gross weight."
        )
        print("'*' buses that do not comply wih energy reduction target.")
        print("'/' vehicles not available for the specified year or charging strategy.")
        print(
            "'O' electric vehicles that do not have enough time to charge batteries overnight."
        )

        # If the number of remaining non-compliant vehicles is not zero, then
        if len([y for y in self.array.year.values if y > 2020]) > 0:

            l_pwt = [
                p for p in self.array.powertrain.values if p in ["ICEV-d", "ICEV-g"]
            ]

            if len(l_pwt) > 0:

                self.array.loc[
                    dict(
                        powertrain=l_pwt,
                        parameter="is_compliant",
                        year=[y for y in self.array.year.values if y > 2020],
                    )
                ] = np.logical_not(non_compliant_vehicles).astype(int)

        # Display of table with passengers onboard
        t = PrettyTable([""] + self.array.coords["size"].values.tolist())

        for pt in self.array.coords["powertrain"].values:
            for y in self.array.coords["year"].values:

                row = [pt + ", " + str(y)]

                vals = []

                for s in self.array.coords["size"].values:

                    # fetch number of passengers onboard
                    val = self.array.sel(
                        parameter=["average passengers"],
                        powertrain=pt,
                        year=y,
                        value="reference"
                        if "reference" in self.array.coords["value"]
                        else 0,
                        size=s,
                    ).values.tolist()

                    # indicate vehicles that do not comply with energy target
                    val = np.where(
                        self.array.sel(
                            parameter="is_compliant",
                            powertrain=pt,
                            year=y,
                            value="reference"
                            if "reference" in self.array.coords["value"]
                            else 0,
                            size=s,
                        ).values,
                        val,
                        [str(v) + "*" for v in val],
                    )

                    # indicate vehicles that have schedule issues
                    val = np.where(
                        self.array.sel(
                            parameter="has_schedule_issue",
                            powertrain=pt,
                            year=y,
                            value="reference"
                            if "reference" in self.array.coords["value"]
                            else 0,
                            size=s,
                        ).values,
                        "O",
                        val,
                    )

                    # indicate vehicles that are too heavy
                    val = np.where(
                        self.array.sel(
                            parameter="is_too_heavy",
                            powertrain=pt,
                            year=y,
                            value="reference"
                            if "reference" in self.array.coords["value"]
                            else 0,
                            size=s,
                        ).values,
                        "X",
                        val,
                    )

                    # indicate vehicles that are not commercially available
                    val = np.where(
                        self.array.sel(
                            parameter="is_available",
                            powertrain=pt,
                            year=y,
                            value="reference"
                            if "reference" in self.array.coords["value"]
                            else 0,
                            size=s,
                        ).values,
                        val,
                        "/",
                    )

                    vals.append(val[0])

                t.add_row(row + vals)
        print(t)

    def adjust_combustion_power_share(self):
        """
        If the exhaust CO2 emissions exceed the targets defined in `self.emission_target`,
        compared to 2020, we decrement the power supply share of the combustion engine.

        :returns: `is_compliant`, whether all vehicles are compliant or not.
        """

        list_target_years = [2020] + list(self.energy_target.keys())
        list_target_vals = [1] + list(self.energy_target.values())
        # years under target
        actual_years = [y for y in self.array.year.values if y > 2020]

        if len(actual_years) > 0:

            l_pwt = [
                p for p in self.array.powertrain.values if p in ["ICEV-d", "ICEV-g"]
            ]

            if len(l_pwt) > 0:

                fc = self.array.loc[:, l_pwt, "TtW energy", :].interp(
                    year=list_target_years, kwargs={"fill_value": "extrapolate"}
                )

                fc[:, :, :, :] = (
                    fc[:, :, 0, :].values
                    * np.array(list_target_vals).reshape(-1, 1, 1, 1)
                ).transpose(1, 2, 0, 3)

                years_after_last_target = [
                    y for y in actual_years if y > list_target_years[-1]
                ]

                list_years = list_target_years + actual_years
                list_years = list(set(list_years))
                fc = fc.interp(year=list_years, kwargs={"fill_value": "extrapolate"})

                if (
                    len(years_after_last_target) > 0
                    and list_target_years[-1] in fc.year.values
                ):

                    fc.loc[dict(year=years_after_last_target)] = fc.loc[
                        dict(year=list_target_years[-1])
                    ].values[:, :, None, :]

                fc = fc.loc[dict(year=actual_years)]

                arr = (
                    fc.values
                    < self.array.loc[:, l_pwt, "TtW energy", actual_years].values
                )

                if arr.sum() > 0:
                    new_shares = (
                        self.array.loc[
                            dict(
                                powertrain=l_pwt,
                                parameter="combustion power share",
                                year=actual_years,
                            )
                        ]
                        - (arr * 0.04)
                    )

                    self.array.loc[
                        dict(
                            powertrain=l_pwt,
                            parameter="combustion power share",
                            year=actual_years,
                        )
                    ] = np.clip(new_shares, 0.6, 1)

                return arr
            else:
                return np.array([])
        else:
            return np.array([])

    def check_compliance_of_buses(self):

        # Indicate vehicles not available before 2020
        # Essentially all BEVs and FCEVs
        l_pwt = [
            p
            for p in self.array.powertrain.values
            if p in ["BEV-opp", "BEV-depot", "FCEV", "HEV-d"]
        ]

        if len(l_pwt) > 0:
            self.array.loc[
                dict(
                    powertrain=l_pwt,
                    parameter="is_available",
                    year=[y for y in self.array.year.values if y < 2020],
                )
            ] = 0

        # Indicates in motion-type and opportunity-type vehicles not available for inter-city purpose
        l_size = [
            s
            for s in self.array.coords["size"].values
            if s in ["13m-coach", "13m-coach-double"]
            and any(
                p
                for p in self.array.powertrain.values
                if p in ["BEV-opp", "BEV-motion"]
            )
        ]

        if len(l_size) > 0:
            self.array.loc[
                dict(
                    powertrain=[
                        p
                        for p in self.array.powertrain.values
                        if p in ["BEV-opp", "BEV-motion"]
                    ],
                    parameter="is_available",
                    size=l_size,
                )
            ] = 0

        # Indicate that in-motion bus are only available for 13m and 18m sizes
        l_size = [
            s
            for s in self.array.coords["size"].values
            if s in ["9m", "13m-city-double"]
            and "BEV-motion" in self.array.powertrain.values
        ]

        if len(l_size) > 0:
            self.array.loc[
                dict(
                    powertrain="BEV-motion",
                    parameter="is_available",
                    size=l_size,
                )
            ] = 0

        # if the mass allowance left
        # if not enough to welcome the average passenger number +50%
        # then the bus is too heavy as undersized for peak times
        self["is_too_heavy"] = np.where(
            np.floor(
                (self["gross mass"] - self["curb mass"])
                / self["average passenger mass"]
            )
            > self["average passengers"] * 1.5,
            0,
            1,
        )

        # check that batteries of plugin BEVs
        # have enough time to charge

        if "BEV-depot" in self.array.powertrain.values:
            with self("BEV-depot") as cpm:
                cpm["has_schedule_issue"] = np.where(
                    (cpm["electric energy stored"] * 1000 / cpm["plugin charger power"])
                    < (24 - cpm["operation time"]),
                    0,
                    1,
                )

    def adjust_cost(self):
        """
        This method adjusts costs of energy storage over time, to correct for the overly optimistic linear
        interpolation between years.

        """

        n_iterations = self.array.shape[-1]
        n_year = len(self.array.year.values)

        # If uncertainty is not considered, teh cost factor equals 1.
        # Otherwise, a variability of +/-30% is added.

        if n_iterations == 1:
            cost_factor = 1

            # reflect a scaling effect for fuel cells
            # according to
            # FCEV trucks should cost the triple of an ICEV-d in 2020
            cost_factor_fcev = 5
        else:
            if "reference" in self.array.value.values:
                cost_factor = np.ones((n_iterations, 1))
                cost_factor_fcev = np.full((n_iterations, 1), 5)
            else:
                cost_factor = np.random.triangular(0.7, 1, 1.3, (n_iterations, 1))
                cost_factor_fcev = np.random.triangular(3, 5, 6, (n_iterations, 1))

        # Correction of hydrogen tank cost, per kg
        if "FCEV" in self.array.powertrain.values:
            self.array.loc[:, ["FCEV"], "fuel tank cost per kg", :, :] = np.reshape(
                (1.078e58 * np.exp(-6.32e-2 * self.array.year.values) + 3.43e2)
                * cost_factor_fcev,
                (1, 1, n_year, n_iterations),
            )

            # Correction of fuel cell stack cost, per kW
            self.array.loc[:, ["FCEV"], "fuel cell cost per kW", :, :] = np.reshape(
                (3.15e66 * np.exp(-7.35e-2 * self.array.year.values) + 2.39e1)
                * cost_factor_fcev,
                (1, 1, n_year, n_iterations),
            )

        # Correction of energy battery system cost, per kWh
        l_pwt = [
            p
            for p in self.array.powertrain.values
            if p in ["BEV-opp", "BEV-depot", "BEV-motion"]
        ]

        if len(l_pwt) > 0:
            self.array.loc[:, l_pwt, "energy battery cost per kWh", :, :,] = np.reshape(
                (2.75e86 * np.exp(-9.61e-2 * self.array.year.values) + 5.059e1)
                * cost_factor,
                (1, 1, n_year, n_iterations),
            )

        # Correction of power battery system cost, per kW
        l_pwt = [
            p
            for p in self.array.powertrain.values
            if p in ["ICEV-d", "ICEV-g", "FCEV", "HEV-d"]
        ]

        if len(l_pwt) > 0:
            self.array.loc[:, l_pwt, "power battery cost per kW", :, :,] = np.reshape(
                (8.337e40 * np.exp(-4.49e-2 * self.array.year.values) + 11.17)
                * cost_factor,
                (1, 1, n_year, n_iterations),
            )

        # Correction of combustion powertrain cost for ICEV-g
        if "ICEV-g" in self.array.powertrain.values:
            self.array.loc[
                :,
                ["ICEV-g"],
                "combustion powertrain cost per kW",
                :,
                :,
            ] = np.reshape(
                (5.92e160 * np.exp(-0.1819 * self.array.year.values) + 26.76)
                * cost_factor,
                (1, 1, n_year, n_iterations),
            )

    def set_electricity_consumption(self):
        """
        This method calculates the total electricity consumption for BEV and plugin-hybrid vehicles
        :returns: Does not return anything. Modifies ``self.array`` in place.
        """
        l_pwt = [
            p
            for p in self.array.powertrain.values
            if p in ["BEV-depot", "BEV-opp", "BEV-motion"]
        ]

        for pt in l_pwt:
            with self(pt) as cpm:
                cpm["electricity consumption"] = (
                    cpm["TtW energy"]
                    / cpm["battery charge efficiency"]
                    / cpm["charger efficiency"]
                ) / 3600

    def calculate_ttw_energy(self):
        """
        This method calculates the energy required to operate auxiliary services as well
        as to move the car. The sum is stored under the parameter label "TtW energy" in :attr:`self.array`.

        """
        self.energy = xr.DataArray(
            np.zeros(
                (
                    len(self.array.coords["size"]),
                    len(self.array.coords["powertrain"]),
                    12,
                    len(self.array.coords["year"]),
                    len(self.array.coords["value"]),
                    self.ecm.cycle.shape[0],
                )
            ).astype("float32"),
            coords=[
                self.array.coords["size"],
                self.array.coords["powertrain"],
                [
                    "auxiliary energy",
                    "heating energy",
                    "cooling energy",
                    "battery management energy",
                    "auxiliary base energy",
                    "motive energy",
                    "motive energy at wheels",
                    "recuperated energy",
                    "recuperated energy at wheels",
                    "transmission efficiency",
                    "engine efficiency",
                    "power load",
                ],
                self.array.coords["year"],
                self.array.coords["value"],
                np.arange(self.ecm.cycle.shape[0]),
            ],
            dims=["size", "powertrain", "parameter", "year", "value", "second"],
        )

        motive_power, recuperated_power, distance = self.ecm.motive_energy_per_km(
            driving_mass=self["driving mass"],
            rr_coef=self["rolling resistance coefficient"],
            drag_coef=self["aerodynamic drag coefficient"],
            frontal_area=self["frontal area"],
            recuperation_efficiency=self["recuperation efficiency"],
            motor_power=self["electric power"],
        )

        self.energy.loc[dict(parameter="motive energy at wheels")] = np.clip(
            motive_power.T, 0, self["power"].values[..., None]
        )

        self.energy.loc[dict(parameter="recuperated energy at wheels")] = (
            np.clip(recuperated_power.T, 0, self["electric power"].values[..., None])
            * -1
        )

        self.energy.loc[dict(parameter="power load")] = (
            motive_power.T + recuperated_power.T
        ) / self["combustion power"].values[..., None]

        (
            p_cooling,
            p_heating,
            p_battery_cooling,
            p_battery_heating,
        ) = self.ecm.aux_energy_per_km(
            hvac_power=self["HVAC power"],
            battery_cooling_unit=self["battery cooling unit"],
            battery_heating_unit=self["battery heating unit"],
        )

        p_cooling = (
            p_cooling[..., None]
            * self["cooling energy consumption"].values[..., None]
            / self["heat pump CoP, cooling"].values[..., None]
        )
        p_heating = (
            p_heating[..., None]
            * self["heating energy consumption"].values[..., None]
            / self["heat pump CoP, heating"].values[..., None]
        )

        p_battery_cooling = np.resize(p_battery_cooling, p_cooling.shape)
        p_battery_heating = np.resize(p_battery_heating, p_cooling.shape)

        self.energy.loc[dict(parameter="heating energy")] = p_heating / 1000
        self.energy.loc[dict(parameter="cooling energy")] = p_cooling / 1000
        self.energy.loc[dict(parameter="battery management energy")] = (
            p_battery_heating + p_battery_cooling
        ) / 1000
        self.energy.loc[dict(parameter="auxiliary base energy")] = (
            self["auxiliary power base demand"].values[..., None] / 1000
        )

        cycle_boundaries = {
            "9m": (7696, 17918),
            "13m-city": (7753, 17918),
            "13m-coach": (16134, 17918),
            "13m-city-double": (7839, 17918),
            "18m": (7965, 17918),
        }

        for b in cycle_boundaries:
            if b in self.array.coords["size"].values:
                self.energy.loc[
                    dict(
                        parameter=[
                            "heating energy",
                            "cooling energy",
                            "battery management energy",
                            "auxiliary base energy",
                        ],
                        size=b,
                        second=np.arange(
                            cycle_boundaries[b][0], cycle_boundaries[b][1]
                        ),
                    )
                ] = 0

        self.energy.loc[dict(parameter="auxiliary energy")] = self.energy.loc[
            dict(
                parameter=[
                    "heating energy",
                    "cooling energy",
                    "battery management energy",
                    "auxiliary base energy",
                ]
            )
        ].sum(dim="parameter")

        self["auxiliary power demand"] = (
            self.energy.loc[dict(parameter="auxiliary energy")].mean(axis=-1)
            * 1000
            / distance.T
        )

        l_pwt = [
            p
            for p in self.array.powertrain.values
            if p in ["BEV-opp", "BEV-depot", "BEV-motion", "FCEV"]
        ]

        idx = [i for i, j in enumerate(self.array.powertrain.values) if j in l_pwt]

        if len(l_pwt) > 0:

            self.energy.loc[dict(parameter="power load", powertrain=l_pwt)] = (
                motive_power.T[:, idx, ...] + recuperated_power.T[:, idx, ...]
            ) / self.array.sel(parameter="electric power", powertrain=l_pwt).values[
                ..., None
            ]

        self.energy.loc[dict(parameter="power load")] = np.clip(
            self.energy.loc[dict(parameter="power load")], 0.0, 1
        )

        self.energy.loc[dict(parameter="transmission efficiency")] = np.clip(
            np.interp(
                self.energy.loc[dict(parameter="power load")],
                [0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                [0, 0.8, 0.83, 0.86, 0.89, 0.89, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            ),
            0.2,
            1,
        )

        l_pwt = [
            p
            for p in self.array.powertrain.values
            if p in ["ICEV-d", "HEV-d", "ICEV-g"]
        ]

        if len(l_pwt) > 0:

            self.energy.loc[
                dict(
                    parameter="engine efficiency",
                    powertrain=l_pwt,
                )
            ] = np.clip(
                np.interp(
                    self.energy.loc[dict(parameter="power load", powertrain=l_pwt)],
                    [0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [
                        0,
                        0.2,
                        0.24,
                        0.28,
                        0.32,
                        0.33,
                        0.34,
                        0.36,
                        0.37,
                        0.38,
                        0.39,
                        0.4,
                    ],
                ),
                0.2,
                1,
            )

            self.energy.loc[
                dict(parameter="engine efficiency", powertrain=l_pwt)
            ] *= 1 / self.array.sel(
                parameter="engine efficiency",
                powertrain=l_pwt,
            )

        # Correction for CNG trucks
        if "ICEV-g" in self.array.powertrain.values:
            self.energy.loc[
                dict(parameter="engine efficiency", powertrain="ICEV-g")
            ] *= 1 - self.array.sel(
                parameter="CNG engine efficiency correction factor",
                powertrain="ICEV-g",
            )

        # Correction for electric motors
        l_pwt = [
            p
            for p in self.array.powertrain.values
            if p in ["BEV-opp", "BEV-depot", "BEV-motion"]
        ]

        if len(l_pwt) > 0:

            self.energy.loc[dict(parameter="engine efficiency", powertrain=l_pwt)] = (
                self.array.sel(parameter="engine efficiency", powertrain=l_pwt)
                * self.array.sel(
                    parameter="battery discharge efficiency", powertrain=l_pwt
                )
            ).expand_dims(dim="x", axis=-1)

        # Correction for fuel cell stack efficiency
        if "FCEV" in self.array.powertrain.values:
            self.energy.loc[
                dict(parameter="engine efficiency", powertrain=["FCEV"])
            ] = (
                self.array.sel(parameter="engine efficiency", powertrain=["FCEV"])
                * self.array.sel(
                    parameter="fuel cell system efficiency", powertrain=["FCEV"]
                )
                * self.array.sel(
                    parameter="battery discharge efficiency", powertrain=["FCEV"]
                )
            ).expand_dims(
                dim="x", axis=-1
            )

        self.energy.loc[dict(parameter="motive energy at wheels")] = np.clip(
            self.energy.loc[dict(parameter="motive energy at wheels")], 0.4, None
        )

        cycle_boundaries = {
            "9m": (7696, 17918),
            "13m-city": (7753, 17918),
            "13m-coach": (16134, 17918),
            "13m-city-double": (7839, 17918),
            "18m": (7965, 17918),
        }

        for b in cycle_boundaries:
            if b in self.array.coords["size"].values:
                self.energy.loc[
                    dict(
                        parameter=["power load", "motive energy at wheels"],
                        size=b,
                        second=np.arange(
                            cycle_boundaries[b][0], cycle_boundaries[b][1]
                        ),
                    )
                ] = 0

        self.energy.loc[dict(parameter="motive energy")] = (
            self.energy.loc[dict(parameter="motive energy at wheels")]
            / self.energy.loc[dict(parameter="transmission efficiency")]
        )

        self.energy.loc[dict(parameter="motive energy")] = np.clip(
            self.energy.loc[dict(parameter="motive energy")],
            0,
            self["power"].values[..., None],
        )
        self.energy.loc[dict(parameter="motive energy")] /= self.energy.loc[
            dict(parameter="engine efficiency")
        ]

        # a round trip from and to the wheels has to be accounted for
        self.energy.loc[dict(parameter="recuperated energy")] = np.clip(
            (
                self.energy.loc[dict(parameter="recuperated energy at wheels")]
                * (
                    self.energy.loc[dict(parameter="transmission efficiency")]
                    * self.energy.loc[dict(parameter="engine efficiency")]
                    * self.array.loc[dict(parameter="battery charge efficiency")]
                    * self.array.loc[dict(parameter="battery discharge efficiency")]
                    * self.energy.loc[dict(parameter="engine efficiency")]
                    * self.energy.loc[dict(parameter="transmission efficiency")]
                )
            ),
            self["electric power"].values[..., None] * -1,
            0,
        )

        self.energy = self.energy.fillna(0)
        self.energy *= np.isfinite(self.energy)

        self["TtW energy"] = (
            self.energy.sel(
                parameter=["motive energy", "auxiliary energy", "recuperated energy"]
            )
            .sum(dim=["second", "parameter"])
            .T
            / distance
        ).T

        self["auxiliary energy"] = (
            self.energy.sel(parameter="auxiliary energy").sum(dim="second").T / distance
        ).T

        self["TtW efficiency"] = np.nanmean(
            np.where(
                self.energy.loc[dict(parameter="power load")] == 0,
                np.nan,
                (
                    self.energy.loc[dict(parameter="transmission efficiency")]
                    * self.energy.loc[dict(parameter="engine efficiency")]
                ),
            ),
            axis=-1,
        )

        pwt = [
            p
            for p in self.array.powertrain.values
            if p not in ["BEV-opp", "BEV-depot", "BEV-motion", "FCEV"]
        ]

        self.array.loc[
            dict(parameter="engine efficiency", powertrain=pwt)
        ] = np.nanmean(
            np.where(
                self.energy.loc[dict(parameter="power load", powertrain=pwt)] == 0,
                np.nan,
                self.energy.loc[dict(parameter="engine efficiency", powertrain=pwt)],
            ),
            axis=-1,
        )

    def set_fuel_cell_parameters(self):
        """
        Specific setup for fuel cells, which are mild hybrids.
        Must be called after :meth:`.set_power_parameters`.
        """

        if "FCEV" in self.array.coords["powertrain"].values:
            with self("FCEV"):
                self["fuel cell system efficiency"] = (
                    self["fuel cell stack efficiency"]
                    / self["fuel cell own consumption"]
                )
                self["fuel cell power share"] = self["fuel cell power share"].clip(
                    min=0, max=1
                )
                self["fuel cell power"] = (
                    self["power"]
                    * self["fuel cell power share"]
                    * self["fuel cell own consumption"]
                )

                # our basic fuel cell mass is based on a car fuel cell with 800 mW/cm2
                # and 1.02 kg/kW
                self["fuel cell stack mass"] = (
                    1.02
                    * self["fuel cell power"]
                    * (800 / self["fuel cell power area density"])
                )
                self["fuel cell ancillary BoP mass"] = (
                    self["fuel cell power"]
                    * self["fuel cell ancillary BoP mass per power"]
                )
                self["fuel cell essential BoP mass"] = (
                    self["fuel cell power"]
                    * self["fuel cell essential BoP mass per power"]
                )

                self["battery power"] = self["fuel cell power"] * (
                    1 - self["fuel cell power share"]
                )
                self["battery cell mass"] = (
                    self["battery power"] / self["battery cell power density"]
                )
                self["battery BoP mass"] = self["battery cell mass"] * (
                    1 - self["battery cell mass share"]
                )

    def set_battery_fuel_cell_replacements(self):
        """
        This methods calculates the number of replacement batteries needed to match the vehicle lifetime.
        Given the chemistry used, the cycle life is known. Given the lifetime kilometers and the kilometers per charge,
        the number of charge cycles can be inferred.

        If the battery lifetime surpasses the vehicle lifetime, 100% of the burden of the battery production
        is allocated to the vehicle. Also, the number of replacement is rounded up. This means that the entirety
        of the battery replacement is allocated to the vehicle (and not to its potential second life).

        """
        # Number of replacement of battery is rounded *up*

        for pt in [
            pwt
            for pwt in ["BEV-opp", "BEV-depot", "BEV-motion"]
            if pwt in self.array.coords["powertrain"].values
        ]:
            with self(pt) as cpm:
                battery_tech_label = (
                    "battery cycle life, " + self.energy_storage["electric"][pt]
                )
                cpm["battery lifetime replacements"] = finite(
                    np.ceil(
                        np.clip(
                            (
                                # number of charge cycles needed divided by the expected cycle life
                                (
                                    cpm["lifetime kilometers"]
                                    * (cpm["TtW energy"] / 3600)
                                )
                                / cpm["electric energy stored"]
                                / cpm[battery_tech_label]
                            )
                            - 1,
                            1,
                            None,
                        )
                    )
                )

        # The number of fuel cell replacements is based on the average distance driven
        # with a set of fuel cells given their lifetime expressed in hours of use.
        # The number is replacement is rounded *up* as we assume no allocation of burden
        # with a second life

        if "FCEV" in self.array.coords["powertrain"].values:
            with self("FCEV") as pt:
                pt["fuel cell lifetime replacements"] = np.ceil(
                    np.clip(
                        (
                            pt["lifetime kilometers"]
                            / pt["average speed"]
                            / pt["fuel cell lifetime hours"]
                        )
                        - 1,
                        0,
                        5,
                    )
                )

    def set_car_masses(self):
        """
        Define ``curb mass``, ``driving mass``, and ``total cargo mass``.

            * `curb mass <https://en.wikipedia.org/wiki/Curb_weight>`__ is the mass of the vehicle and fuel, without people or cargo.
            * ``total cargo mass`` is the mass of the cargo and passengers.
            * ``driving mass`` is the ``curb mass`` plus ``total cargo mass``.

        .. note::
            driving mass = total cargo mass + driving mass

        """

        # Base components, common to all powertrains
        base_components = [
            "glider base mass",
            "suspension mass",
            "braking system mass",
            "wheels and tires mass",
            "electrical system mass",
            "transmission mass",
            "other components mass",
        ]

        self["curb mass"] = self[base_components].sum(axis=2) * (
            1 - self["lightweighting"]
        )

        curb_mass_includes = [
            "fuel mass",
            "charger mass",
            "converter mass",
            "inverter mass",
            "power distribution unit mass",
            # Updates with set_components_mass
            "combustion engine mass",
            # Updates with set_components_mass
            "electric engine mass",
            # Updates with set_components_mass
            "exhaust system mass",
            "fuel cell stack mass",
            "fuel cell ancillary BoP mass",
            "fuel cell essential BoP mass",
            "battery cell mass",
            "battery BoP mass",
            "fuel tank mass",
            "HVAC mass",
        ]
        self["curb mass"] += self[curb_mass_includes].sum(axis=2)

        # the curb mass is limited by
        # the fact that the mass
        # of the vehicle is full occupied
        # should be under
        # the maximum allowed gross mass

        self["driving mass"] = self["curb mass"] + (
            (self["average passengers"] * self["average passenger mass"])
            + (self["average passengers"] * self["passenger luggage mass"])
        )

    def set_power_parameters(self):
        """Set electric and combustion motor powers based on input parameter ``power to mass ratio``."""
        # Convert from W/kg to kW
        self["power"] = np.clip(
            self["power to mass ratio"] * self["curb mass"] / 1000, 0, 500
        )
        self["combustion power share"] = self["combustion power share"].clip(
            min=0, max=1
        )
        self["combustion power"] = self["power"] * self["combustion power share"]
        self["electric power"] = self["power"] * (1 - self["combustion power share"])

    def set_component_masses(self):
        self["combustion engine mass"] = (
            self["combustion power"] * self["engine mass per power"]
            + self["engine fixed mass"]
        )
        self["electric engine mass"] = np.clip(
            (24.56 * np.exp(0.0078 * self["electric power"])), 0, 600
        ) * (self["combustion power share"] < 1)

        self["transmission mass"] = (self["gross mass"] / 1000) * self[
            "transmission mass per ton of gross weight"
        ] + self["transmission fixed mass"]

        self["inverter mass"] = (
            self["electric power"] * self["inverter mass per power"]
            + self["inverter fix mass"]
        )

        self["HVAC mass"] = (self["HVAC power"] / 1000) * self[
            "HVAC system mass per kW"
        ] + self["HVAC system fixed mass"]

    def set_battery_size(self):

        if "BEV-opp" in self.array.powertrain.values:
            with self("BEV-opp") as cpm:

                # battery sizing factor
                # for fast-charge batteries
                # we operate only between a limited DoD window
                # to prevent cell degradation
                # meaning we use only a specific capacity window
                # hence, the size factor is 1 / (maximum SoC - maximum DoD)
                sizing_factor = 1 / (cpm["maximum SoC"] - cpm["battery DoD"])

                # calculate the number of trips
                # which equals the total daily distance
                # divided by the length of one trip

                # calculate the number of charging opportunities

                cpm["electric energy stored"] = (
                    cpm["distance per trip"]
                    * (cpm["TtW energy"] / 1000)
                    * sizing_factor
                    / 3.6
                    / cpm["charging opportunity per trip"]
                )

        if "BEV-depot" in self.array.powertrain.values:
            # we add a 20% capacity margin
            with self("BEV-depot") as cpm:
                cpm["electric energy stored"] = (
                    cpm["daily distance"]
                    * (cpm["TtW energy"] / 1000)
                    * 1.2
                    / cpm["battery DoD"]
                    / 3.6
                )

            # for trolley BEV buses, it is more complex
            # as the size of the battery depends on the
            # extent of the presence of catenary lines
            # so we use the following rule of thumb:
            # the battery capacity is a third of that
            # of opportunity charging BEV buses

        if "BEV-motion" in self.array.powertrain.values:
            with self("BEV-motion") as cpm:

                # battery sizing factor
                # for fast-charge batteries
                # we operate only between a limited DoD window
                # to prevent cell degradation
                # meaning we use only a specific capacity window
                # hence, the size factor is 1 / (maximum SoC - maximum DoD)
                sizing_factor = 1 / (cpm["maximum SoC"] - cpm["battery DoD"])

                # calculate the number of trips
                # which equals the total daily distance
                # divided by the length of one trip

                # we assume that catenary systems can charge the
                # equivalent of 1kWh / min
                # according to the Zeus project https://www.vdv.de/zeeus-ebus-report-internet.pdfx

                # TtW energy [kj/km] - the charging capacity of the catenary system
                # * the share of the trip equipped with catenary
                # gives us the deficit (battery depletion) per km
                # in Givisiez, around 28% the trip distance is equipped with catenary

                # we add to this a sizing factor
                # to ensure tha the battery is always being charged
                # at a SoC between 40% and 80%
                # to limit degradation

                cpm["electric energy stored"] = np.clip(
                    (
                        (cpm["TtW energy"] / 3600)
                        - (
                            cpm["trip distance share with catenary"]  # km/km
                            / cpm["average speed"]  # km/h
                            * cpm["catenary system power"]
                            / 1000  # kW)
                        )
                    )
                    * sizing_factor
                    * cpm["use of auxiliary battery"]
                    * cpm["distance per trip"],
                    0,
                    None,
                )

        # Fuel cell buses do also have a battery, which capacity
        # corresponds roughly to 9% of the capacity contained in the
        # H2 tank
        if "FCEV" in self.array.powertrain.values:
            with self("FCEV") as cpm:
                cpm["electric energy stored"] = cpm["fuel mass"] * 120 / 3.6 * 0.09

        for pt in [
            pwt
            for pwt in ["BEV-opp", "BEV-depot", "BEV-motion", "FCEV"]
            if pwt in self.array.coords["powertrain"].values
        ]:
            with self(pt) as cpm:

                battery_tech_label = (
                    "battery cell energy density, "
                    + self.energy_storage["electric"][pt]
                )
                cpm["battery cell mass"] = (
                    cpm["electric energy stored"] / cpm[battery_tech_label]
                )

                cpm["energy battery mass"] = (
                    cpm["battery cell mass"] / cpm["battery cell mass share"]
                )

                cpm["battery BoP mass"] = (
                    cpm["energy battery mass"] - cpm["battery cell mass"]
                )

        # kWh electricity/kg battery cell
        self["battery cell production energy electricity share"] = self[
            "battery cell production energy electricity share"
        ].clip(min=0, max=1)
        self["battery cell production electricity"] = (
            self["battery cell production energy"]
            * self["battery cell production energy electricity share"]
        )
        # MJ heat/kg battery cell
        self["battery cell production heat"] = (
            self["battery cell production energy"]
            - self["battery cell production electricity"]
        ) * 3.6

    def set_energy_stored_properties(self):
        """
        First, fuel mass is defined. It is dependent on the range required.
        Then batteries are sized, depending on the range required and the energy consumption.
        :return:
        """
        d_map_fuel = {
            "ICEV-d": "diesel",
            "HEV-d": "diesel",
            "ICEV-g": "cng",
            "FCEV": "hydrogen",
        }

        # average speed along the driving cycle
        cycle = self.cycle.copy()
        velocity = (self.cycle * 1000) / 3600
        self["distance per trip"] = (velocity.sum(axis=0) / 1000)[:, None, None, None]

        # distance driven per day is the product of the daily operation time
        # and the average speed across the driving cycle
        # (we omitted the stops in the average)
        cycle[cycle == 0] = np.nan
        self["average speed"] = np.nanmean(cycle, axis=0)[:, None, None, None]
        self["daily distance"] = self["operation time"].values * self["average speed"]

        # the number of trips is simply the daily driven distance
        # divided by the distance of one trip
        self["number of trips"] = self["daily distance"] / self["distance per trip"]

        for pt in [
            pwt
            for pwt in ["ICEV-d", "HEV-d", "ICEV-g"]
            if pwt in self.array.coords["powertrain"].values
        ]:

            with self(pt) as cpm:

                # calculate the average LHV based on fuel blend
                fuel_type = d_map_fuel[pt]
                primary_fuel_share = self.fuel_blend[fuel_type]["primary"]["share"]
                primary_fuel_lhv = self.fuel_blend[fuel_type]["primary"]["lhv"]
                secondary_fuel_share = self.fuel_blend[fuel_type]["secondary"]["share"]
                secondary_fuel_lhv = self.fuel_blend[fuel_type]["secondary"]["lhv"]

                blend_lhv = (np.array(primary_fuel_share) * primary_fuel_lhv) + (
                    np.array(secondary_fuel_share) * secondary_fuel_lhv
                )

                cpm["fuel mass"] = (
                    cpm["daily distance"] * (cpm["TtW energy"] / 1000)
                ) / blend_lhv.reshape(-1, 1)

                cpm["oxidation energy stored"] = (
                    cpm["fuel mass"] * blend_lhv.reshape(-1, 1)
                ) / 3.6

                if pt == "ICEV-g":
                    # Based on manufacturer data
                    # We use a four-cyclinder configuration
                    # Of 320L each
                    # a_matrix cylinder of 320L @ 200 bar can hold 57.6 kg of CNG
                    nb_cylinder = np.ceil(cpm["fuel mass"] / 57.6)

                    cpm["fuel tank mass"] = (
                        (0.018 * np.power(57.6, 2)) - (0.6011 * 57.6) + 52.235
                    ) * nb_cylinder

                else:

                    # From Wolff et al. 2020, Sustainability, DOI: 10.3390/su12135396.
                    # We adjusted though the intercept from the original function (-54)
                    # because we size here trucks based on the range autonomy
                    # a low range autonomy would produce a negative fuel tank mass
                    cpm["fuel tank mass"] = (
                        17.159 * np.log(cpm["fuel mass"] * (1 / 0.832)) - 30
                    )

        for pt in [
            pwt
            for pwt in ["ICEV-d", "HEV-d", "ICEV-g"]
            if pwt in self.array.coords["powertrain"].values
        ]:
            with self(pt) as cpm:
                cpm["battery power"] = cpm["electric power"]
                cpm["battery cell mass"] = (
                    cpm["battery power"] / cpm["battery cell power density"]
                )
                cpm["battery cell mass share"] = cpm["battery cell mass share"].clip(
                    min=0, max=1
                )
                cpm["battery BoP mass"] = cpm["battery cell mass"] * (
                    1 - cpm["battery cell mass share"]
                )

        if "FCEV" in self.array.coords["powertrain"].values:
            with self("FCEV") as cpm:
                cpm["fuel mass"] = (
                    cpm["daily distance"] * (cpm["TtW energy"] / 1000)
                ) / cpm["LHV fuel MJ per kg"]
                cpm["oxidation energy stored"] = cpm["fuel mass"] * 120 / 3.6  # kWh

                cpm["fuel tank mass"] = (
                    (-0.1916 * np.power(cpm["fuel mass"], 2))
                    + (14.586 * cpm["fuel mass"])
                    + 10.805
                )

        # for older trolley BEV buses,
        # the share driven without the overhead
        # contact lines require a diesel generator

        if "BEV-motion" in self.array.coords["powertrain"].values:
            with self("BEV-motion") as cpm:

                cpm["oxidation energy stored"] = np.clip(
                    (
                        (cpm["TtW energy"] / 3600)
                        * (1 - cpm["trip distance share with catenary"])
                    )
                    / cpm["generator efficiency"]
                    * cpm["daily distance"]
                    * (1 - cpm["use of auxiliary battery"]),
                    0,
                    None,
                )

    def set_costs(self):

        glider_components = [
            "glider base mass",
            "suspension mass",
            "braking system mass",
            "wheels and tires mass",
        ]

        self["glider cost"] = np.clip(
            ((38747 * np.log(self[glider_components].sum(dim="parameter"))) - 252194),
            33500,
            110000,
        )

        # Discount glider cost for 40t and 60t trucks because of the added trailer mass

        for size in [
            s for s in ["40t", "60t"] if s in self.array.coords["size"].values
        ]:
            self.array.loc[dict(parameter="glider cost", size=size)] *= 0.7

        self["lightweighting cost"] = (
            self["glider base mass"]
            * self["lightweighting"]
            * self["glider lightweighting cost per kg"]
        )
        self["electric powertrain cost"] = (
            self["electric powertrain cost per kW"] * self["electric power"]
        )
        self["combustion powertrain cost"] = (
            self["combustion power"] * self["combustion powertrain cost per kW"]
        )

        self["fuel cell cost"] = self["fuel cell power"] * self["fuel cell cost per kW"]

        self["power battery cost"] = (
            self["battery power"] * self["power battery cost per kW"]
        )
        self["energy battery cost"] = (
            self["energy battery cost per kWh"] * self["electric energy stored"]
        )
        self["fuel tank cost"] = self["fuel tank cost per kg"] * self["fuel mass"]
        # Per passenger-km
        self["energy cost"] = (
            self["energy cost per kWh"]
            * self["TtW energy"]
            / 3600
            / self["average passengers"]
        )

        # For battery, need to divide cost of electricity in battery by efficiency of charging
        for pt in [
            pwt
            for pwt in [
                "BEV-opp",
                "BEV-depot",
                "BEV-motion",
            ]
            if pwt in self.array.coords["powertrain"].values
        ]:
            with self(pt) as cpm:
                cpm["energy cost"] /= cpm["battery charge efficiency"]

        self["component replacement cost"] = (
            self["energy battery cost"] * self["battery lifetime replacements"]
            + self["fuel cell cost"] * self["fuel cell lifetime replacements"]
        )

        to_markup = [
            "combustion powertrain cost",
            "component replacement cost",
            "electric powertrain cost",
            "energy battery cost",
            "fuel cell cost",
            "fuel tank cost",
            "glider cost",
            "lightweighting cost",
            "power battery cost",
        ]

        self[to_markup] *= self["markup factor"]

        # calculate costs per km:
        lifetime = self["lifetime kilometers"] / self["kilometers per year"]
        i = self["interest rate"]
        amortisation_factor = ne.evaluate("i + (i / ((1 + i) ** lifetime - 1))")

        purchase_cost_list = [
            "battery onboard charging infrastructure cost",
            "combustion exhaust treatment cost",
            "combustion powertrain cost",
            "electric powertrain cost",
            "energy battery cost",
            "fuel cell cost",
            "fuel tank cost",
            "glider cost",
            "heat pump cost",
            "lightweighting cost",
            "power battery cost",
        ]

        self["purchase cost"] = self[purchase_cost_list].sum(axis=2)

        # per passenger-km
        self["amortised purchase cost"] = (
            self["purchase cost"]
            * amortisation_factor
            / self["average passengers"]
            / self["kilometers per year"]
        )

        # per passenger-km
        self["adblue cost"] = (
            self["adblue cost per kg"] * 0.06 * self["fuel mass"]
        ) / self["daily distance"]
        self["maintenance cost"] = self["maintenance cost per km"]
        self["maintenance cost"] += self["adblue cost"]
        self["maintenance cost"] /= self["average passengers"]

        self["insurance cost"] = (
            self["insurance cost per year"]
            / self["average passengers"]
            / self["kilometers per year"]
        )

        self["toll cost"] = self["toll cost per km"] / self["average passengers"]

        # simple assumption that component replacement occurs at half of life.
        km_per_year = self["kilometers per year"]
        com_repl_cost = self["component replacement cost"]
        passengers = self["average passengers"]

        self["amortised component replacement cost"] = ne.evaluate(
            "(com_repl_cost * ((1 - i) ** lifetime / 2) * amortisation_factor) / km_per_year / passengers"
        )

        self["total cost per km"] = (
            self["energy cost"]
            + self["amortised purchase cost"]
            + self["maintenance cost"]
            + self["insurance cost"]
            + self["toll cost"]
            + self["amortised component replacement cost"]
        )

    def set_particulates_emission(self):
        """
        Calculate the emission of particulates according to
        https://www.eea.europa.eu/ds_resolveuid/6USNA27I4D

        for:

        - brake wear
        - tire wear
        - road wear

        Tier-2 method considers:

        - number of axles
        - capacity utilization rate
        - speed

        """

        list_param = [
            "tire wear emissions",
            "brake wear emissions",
            "road wear emissions",
        ]

        pem = ParticulatesEmissionsModel(
            cycle=self.ecm.cycle,
            number_axles=self["number of axles"],
            load_factor=self["average passengers"]
            / self["initial passengers capacity"],
        )

        res = pem.get_abrasion_emissions()
        self[list_param] = res.transpose(4, 3, 0, 2, 1)

    def set_hot_emissions(self):
        """
        Calculate hot pollutant emissions based on ``driving cycle``.
        The driving cycle is passed to the :class:`HotEmissionsModel` class and :meth:`get_emissions_per_powertrain`
        return emissions per substance per second of driving cycle.
        :return: Does not return anything. Modifies ``self.array`` in place.
        """
        hem = HotEmissionsModel(self.ecm.cycle)

        list_direct_emissions = [
            "Hydrocarbons direct emissions, urban",
            "Carbon monoxide direct emissions, urban",
            "Nitrogen oxides direct emissions, urban",
            "Particulate matters direct emissions, urban",
            "Methane direct emissions, urban",
            "NMVOC direct emissions, urban",
            "Dinitrogen oxide direct emissions, urban",
            "Ammonia direct emissions, urban",
            "Benzene direct emissions, urban",
            "Ethane direct emissions, urban",
            "Propane direct emissions, urban",
            "Butane direct emissions, urban",
            "Pentane direct emissions, urban",
            "Hexane direct emissions, urban",
            "Cyclohexane direct emissions, urban",
            "Heptane direct emissions, urban",
            "Ethene direct emissions, urban",
            "Propene direct emissions, urban",
            "1-Pentene direct emissions, urban",
            "Toluene direct emissions, urban",
            "m-Xylene direct emissions, urban",
            "o-Xylene direct emissions, urban",
            "Formaldehyde direct emissions, urban",
            "Acetaldehyde direct emissions, urban",
            "Benzaldehyde direct emissions, urban",
            "Acetone direct emissions, urban",
            "Methyl ethyl ketone direct emissions, urban",
            "Acrolein direct emissions, urban",
            "Styrene direct emissions, urban",
            "PAH, polycyclic aromatic hydrocarbons direct emissions, urban",
            "Arsenic direct emissions, urban",
            "Selenium direct emissions, urban",
            "Zinc direct emissions, urban",
            "Copper direct emissions, urban",
            "Nickel direct emissions, urban",
            "Chromium direct emissions, urban",
            "Chromium VI direct emissions, urban",
            "Mercury direct emissions, urban",
            "Cadmium direct emissions, urban",
            "Hydrocarbons direct emissions, suburban",
            "Carbon monoxide direct emissions, suburban",
            "Nitrogen oxides direct emissions, suburban",
            "Particulate matters direct emissions, suburban",
            "Methane direct emissions, suburban",
            "NMVOC direct emissions, suburban",
            "Dinitrogen oxide direct emissions, suburban",
            "Ammonia direct emissions, suburban",
            "Benzene direct emissions, suburban",
            "Ethane direct emissions, suburban",
            "Propane direct emissions, suburban",
            "Butane direct emissions, suburban",
            "Pentane direct emissions, suburban",
            "Hexane direct emissions, suburban",
            "Cyclohexane direct emissions, suburban",
            "Heptane direct emissions, suburban",
            "Ethene direct emissions, suburban",
            "Propene direct emissions, suburban",
            "1-Pentene direct emissions, suburban",
            "Toluene direct emissions, suburban",
            "m-Xylene direct emissions, suburban",
            "o-Xylene direct emissions, suburban",
            "Formaldehyde direct emissions, suburban",
            "Acetaldehyde direct emissions, suburban",
            "Benzaldehyde direct emissions, suburban",
            "Acetone direct emissions, suburban",
            "Methyl ethyl ketone direct emissions, suburban",
            "Acrolein direct emissions, suburban",
            "Styrene direct emissions, suburban",
            "PAH, polycyclic aromatic hydrocarbons direct emissions, suburban",
            "Arsenic direct emissions, suburban",
            "Selenium direct emissions, suburban",
            "Zinc direct emissions, suburban",
            "Copper direct emissions, suburban",
            "Nickel direct emissions, suburban",
            "Chromium direct emissions, suburban",
            "Chromium VI direct emissions, suburban",
            "Mercury direct emissions, suburban",
            "Cadmium direct emissions, suburban",
            "Hydrocarbons direct emissions, rural",
            "Carbon monoxide direct emissions, rural",
            "Nitrogen oxides direct emissions, rural",
            "Particulate matters direct emissions, rural",
            "Methane direct emissions, rural",
            "NMVOC direct emissions, rural",
            "Dinitrogen oxide direct emissions, rural",
            "Ammonia direct emissions, rural",
            "Benzene direct emissions, rural",
            "Ethane direct emissions, rural",
            "Propane direct emissions, rural",
            "Butane direct emissions, rural",
            "Pentane direct emissions, rural",
            "Hexane direct emissions, rural",
            "Cyclohexane direct emissions, rural",
            "Heptane direct emissions, rural",
            "Ethene direct emissions, rural",
            "Propene direct emissions, rural",
            "1-Pentene direct emissions, rural",
            "Toluene direct emissions, rural",
            "m-Xylene direct emissions, rural",
            "o-Xylene direct emissions, rural",
            "Formaldehyde direct emissions, rural",
            "Acetaldehyde direct emissions, rural",
            "Benzaldehyde direct emissions, rural",
            "Acetone direct emissions, rural",
            "Methyl ethyl ketone direct emissions, rural",
            "Acrolein direct emissions, rural",
            "Styrene direct emissions, rural",
            "PAH, polycyclic aromatic hydrocarbons direct emissions, rural",
            "Arsenic direct emissions, rural",
            "Selenium direct emissions, rural",
            "Zinc direct emissions, rural",
            "Copper direct emissions, rural",
            "Nickel direct emissions, rural",
            "Chromium direct emissions, rural",
            "Chromium VI direct emissions, rural",
            "Mercury direct emissions, rural",
            "Cadmium direct emissions, rural",
        ]

        l_y = []
        for y in self.array.year.values:
            if 2000 <= y < 2005:
                l_y.append(3)
            if 2005 <= y < 2008:
                l_y.append(4)
            if 2008 <= y < 2013:
                l_y.append(5)
            if y >= 2013:
                l_y.append(6)

        l_pwt = [p for p in self.array.powertrain.values if p in ["ICEV-d", "HEV-d"]]

        if len(l_pwt) > 0:
            self.array.loc[
                dict(
                    powertrain=l_pwt,
                    parameter=list_direct_emissions,
                )
            ] = hem.get_emissions_per_powertrain(
                powertrain_type="diesel",
                euro_classes=l_y,
                lifetime_km=self.array.loc[
                    dict(
                        powertrain=l_pwt,
                        parameter="lifetime kilometers",
                    )
                ],
                energy_consumption=self.energy.sel(
                    powertrain=l_pwt,
                    parameter=[
                        "motive energy",
                        "auxiliary energy",
                        "recuperated energy",
                    ],
                ).sum(dim="parameter"),
                size=self.array.coords["size"].values,
            )

        # For CNG vehicles
        if "ICEV-g" in self.array.powertrain.values:
            self.array.loc[
                dict(powertrain="ICEV-g", parameter=list_direct_emissions)
            ] = np.squeeze(
                hem.get_emissions_per_powertrain(
                    powertrain_type="cng",
                    euro_classes=l_y,
                    lifetime_km=self.array.loc[
                        dict(
                            powertrain="ICEV-g",
                            parameter="lifetime kilometers",
                        )
                    ],
                    energy_consumption=self.energy.sel(
                        powertrain=["ICEV-g"],
                        parameter=[
                            "motive energy",
                            "auxiliary energy",
                            "recuperated energy",
                        ],
                    ).sum(dim="parameter"),
                    size=self.array.coords["size"].values,
                ),
                axis=1,
            )

    def set_noise_emissions(self):
        """
        Calculate noise emissions based on ``driving cycle``.
        The driving cycle is passed to the :class:`NoiseEmissionsModel` class and :meth:`get_sound_power_per_compartment`
        returns emissions per compartment type ("rural", "non-urban" and "urban") per second of driving cycle.

        Noise emissions are not differentiated by size classes at the moment, but only by powertrain "type"
        (e.g., combustion, hybrid and electric)

        :return: Does not return anything. Modifies ``self.array`` in place.
        """
        nem = NoiseEmissionsModel()

        list_noise_emissions = [
            "noise, octave 1, day time, urban",
            "noise, octave 2, day time, urban",
            "noise, octave 3, day time, urban",
            "noise, octave 4, day time, urban",
            "noise, octave 5, day time, urban",
            "noise, octave 6, day time, urban",
            "noise, octave 7, day time, urban",
            "noise, octave 8, day time, urban",
            "noise, octave 1, day time, suburban",
            "noise, octave 2, day time, suburban",
            "noise, octave 3, day time, suburban",
            "noise, octave 4, day time, suburban",
            "noise, octave 5, day time, suburban",
            "noise, octave 6, day time, suburban",
            "noise, octave 7, day time, suburban",
            "noise, octave 8, day time, suburban",
            "noise, octave 1, day time, rural",
            "noise, octave 2, day time, rural",
            "noise, octave 3, day time, rural",
            "noise, octave 4, day time, rural",
            "noise, octave 5, day time, rural",
            "noise, octave 6, day time, rural",
            "noise, octave 7, day time, rural",
            "noise, octave 8, day time, rural",
        ]

        l_pwt_combustion = [
            p
            for p in self.array.powertrain.values
            if p
            in [
                "ICEV-g",
                "ICEV-d",
            ]
        ]

        l_pwt_electric = [
            p
            for p in self.array.powertrain.values
            if p
            in [
                "BEV-opp",
                "BEV-depot",
                "BEV-motion",
                "FCEV",
            ]
        ]

        l_size_medium = [
            s
            for s in self.array.coords["size"].values
            if s in ["9m", "13m-city", "13m-coach"]
        ]

        l_size_heavy = [
            s
            for s in self.array.coords["size"].values
            if s in ["13m-city-double", "13m-coach-double", "18m"]
        ]

        if len(l_pwt_combustion) > 0 and len(l_size_medium) > 0:
            cycle = get_standard_driving_cycle(size=l_size_medium)

            self.array.loc[
                dict(
                    powertrain=l_pwt_combustion,
                    parameter=list_noise_emissions,
                    size=l_size_medium,
                )
            ] = nem.get_sound_power_per_compartment(
                powertrain_type="combustion",
                category="medium",
                cycle=cycle,
                size=l_size_medium,
            )

        if len(l_pwt_combustion) > 0 and len(l_size_heavy) > 0:
            cycle = get_standard_driving_cycle(size=l_size_heavy)

            self.array.loc[
                dict(
                    powertrain=l_pwt_combustion,
                    parameter=list_noise_emissions,
                    size=l_size_heavy,
                )
            ] = nem.get_sound_power_per_compartment(
                powertrain_type="combustion",
                category="heavy",
                cycle=cycle,
                size=l_size_heavy,
            )

        if len(l_pwt_electric) > 0 and len(l_size_medium) > 0:
            cycle = get_standard_driving_cycle(size=l_size_medium)

            self.array.loc[
                dict(
                    powertrain=l_pwt_electric,
                    parameter=list_noise_emissions,
                    size=l_size_medium,
                )
            ] = nem.get_sound_power_per_compartment(
                powertrain_type="electric",
                category="medium",
                cycle=cycle,
                size=l_size_medium,
            )

        if len(l_pwt_electric) > 0 and len(l_size_heavy) > 0:
            cycle = get_standard_driving_cycle(size=l_size_heavy)

            self.array.loc[
                dict(
                    powertrain=l_pwt_electric,
                    parameter=list_noise_emissions,
                    size=l_size_heavy,
                )
            ] = nem.get_sound_power_per_compartment(
                powertrain_type="electric",
                category="heavy",
                cycle=cycle,
                size=l_size_heavy,
            )

        if "HEV-d" in self.array.powertrain.values and len(l_size_medium) > 0:
            cycle = get_standard_driving_cycle(size=l_size_medium)

            self.array.loc[
                dict(
                    powertrain=["HEV-d"],
                    parameter=list_noise_emissions,
                    size=l_size_medium,
                )
            ] = nem.get_sound_power_per_compartment(
                powertrain_type="hybrid",
                category="medium",
                cycle=cycle,
                size=l_size_medium,
            )

        if "HEV-d" in self.array.powertrain.values and len(l_size_heavy) > 0:
            cycle = get_standard_driving_cycle(size=l_size_heavy)

            self.array.loc[
                dict(
                    powertrain=["HEV-d"],
                    parameter=list_noise_emissions,
                    size=l_size_heavy,
                )
            ] = nem.get_sound_power_per_compartment(
                powertrain_type="hybrid",
                category="heavy",
                cycle=cycle,
                size=l_size_heavy,
            )

    def calculate_cost_impacts(self, sensitivity=False, scope=None):
        """
        This method returns an array with cost values per vehicle-km, sub-divided into the following groups:

            * Purchase
            * Maintentance
            * Component replacement
            * Energy
            * Total cost of ownership

        :return: a_matrix xarray array with cost information per vehicle-km
        :rtype: xarray.core.dataarray.DataArray
        """

        if scope is None:
            scope = {
                "size": self.array.coords["size"].values.tolist(),
                "powertrain": self.array.coords["powertrain"].values.tolist(),
                "year": self.array.coords["year"].values.tolist(),
            }
        else:
            scope["size"] = scope.get("size", self.array.coords["size"].values.tolist())
            scope["powertrain"] = scope.get(
                "powertrain", self.array.coords["powertrain"].values.tolist()
            )
            scope["year"] = scope.get("year", self.array.coords["year"].values.tolist())

        list_cost_cat = [
            "purchase",
            "maintenance",
            "insurance",
            "toll",
            "component replacement",
            "energy",
            "total",
        ]

        response = xr.DataArray(
            np.zeros(
                (
                    len(scope["size"]),
                    len(scope["powertrain"]),
                    len(list_cost_cat),
                    len(scope["year"]),
                    len(self.array.coords["value"].values),
                )
            ),
            coords=[
                scope["size"],
                scope["powertrain"],
                list_cost_cat,
                scope["year"],
                self.array.coords["value"].values.tolist(),
            ],
            dims=["size", "powertrain", "cost_type", "year", "value"],
        )

        response.loc[:, :, list_cost_cat, :, :,] = self.array.sel(
            powertrain=scope["powertrain"],
            size=scope["size"],
            year=scope["year"],
            parameter=[
                "amortised purchase cost",
                "maintenance cost",
                "insurance cost",
                "toll cost",
                "amortised component replacement cost",
                "energy cost",
                "total cost per km",
            ],
        ).values

        if not sensitivity:
            return response * (self.array.sel(parameter="total cargo mass") > 100)
        else:
            return response / response.sel(value="reference")

    def get_share_biofuel(self):

        try:
            region = self.bs.region_map[self.country]["RegionCode"]
        except KeyError:
            print(
                "No biofuel share could be found for {}. Hence, EU average is used.".format(
                    self.country
                )
            )
            region = "EUR"
        scenario = "SSP2-Base"

        share_biofuel = (
            self.bs.biofuel.sel(
                region=region,
                value=0,
                fuel_type="Biomass fuel",
                scenario=scenario,
            )
            .interp(
                year=self.array.coords["year"].values,
                kwargs={"fill_value": "extrapolate"},
            )
            .values
        )
        return share_biofuel

    def find_fuel_shares(self, fuel_blend, fuel_type):

        default_fuels = {
            "diesel": {
                "primary": "diesel",
                "secondary": "biodiesel - cooking oil",
                "all": [
                    "diesel",
                    "biodiesel - cooking oil",
                    "biodiesel - algae",
                    "biodiesel - palm oil",
                    "biodiesel - rapeseed oil",
                    "synthetic diesel - energy allocation",
                ],
            },
            "cng": {
                "primary": "cng",
                "secondary": "biogas - sewage sludge",
                "all": ["cng", "biogas - sewage sludge", "syngas", "biogas - biowaste"],
            },
            "hydrogen": {
                "primary": "electrolysis",
                "secondary": "smr - natural gas",
                "all": [
                    "electrolysis",
                    "smr - natural gas",
                    "smr - natural gas with CCS",
                    "smr - biogas",
                    "smr - biogas with CCS",
                    "coal gasification",
                    "wood gasification",
                    "wood gasification with CCS",
                ],
            },
        }

        if fuel_type in fuel_blend:
            primary = fuel_blend[fuel_type]["primary"]["type"]

            try:
                # See of a secondary fuel type has been specified
                secondary = fuel_blend[fuel_type]["secondary fuel"]["type"]
            except:
                # a_matrix secondary fuel has not been specified, set one by default
                # Check first if the default fuel is not similar to the primary fuel

                if default_fuels[fuel_type]["secondary"] != primary:
                    secondary = default_fuels[fuel_type]["secondary"]
                else:
                    secondary = [
                        f for f in default_fuels[fuel_type]["all"] if f != primary
                    ][0]

            primary_share = fuel_blend[fuel_type]["primary"]["share"]
            secondary_share = 1 - np.array(primary_share)

        else:
            primary = default_fuels[fuel_type]["primary"]
            secondary = default_fuels[fuel_type]["secondary"]

            if primary == "electrolysis":
                secondary_share = np.zeros_like(self.array.year.values)
            else:
                secondary_share = self.get_share_biofuel()

            primary_share = 1 - np.array(secondary_share)

        return primary, secondary, primary_share, secondary_share

    def define_fuel_blends(self, fuel_blend=None):
        """
        This function defines fuel blends from what is passed in `fuel_blend`.
        It populates a dictionary `self.fuel_blends` that contains the respective shares, lower heating values
        and CO2 emission factors of the fuels used.

        Source for LHV: https://www.bafu.admin.ch/bafu/en/home/topics/climate/state/data/climate-reporting/references.html
        :return:
        """
        # MJ/kg
        fuels_lhv = {
            "diesel": 43,
            "biodiesel - cooking oil": 38,
            "biodiesel - palm oil": 38,
            "biodiesel - rapeseed oil": 38,
            "biodiesel - algae": 38,
            "synthetic diesel - economic allocation": 43,
            "synthetic diesel - energy allocation": 43,
            "cng": 47.5,
            "lpg": 49.3,
            "biogas - sewage sludge": 47.5,
            "biogas - biowaste": 47.5,
            "syngas": 47.5,
            "electrolysis": 120,
            "smr - natural gas": 120,
            "smr - natural gas with CCS": 120,
            "smr - biogas": 120,
            "smr - biogas with CCS": 120,
            "atr - natural gas": 120,
            "atr - natural gas with CCS": 120,
            "atr - biogas": 120,
            "atr - biogas with CCS": 120,
            "coal gasification": 120,
            "wood gasification": 120,
            "wood gasification (Swiss forest)": 120,
            "wood gasification with CCS": 120,
            "wood gasification with CCS (Swiss forest)": 120,
            "wood gasification with EF": 120,
            "wood gasification with EF (Swiss forest)": 120,
            "wood gasification with EF with CCS": 120,
            "wood gasification with EF with CCS (Swiss forest)": 120,
        }

        fuels_CO2 = {
            "diesel": 3.15,
            "biodiesel - cooking oil": 2.79,
            "biodiesel - algae": 2.79,
            "biodiesel - palm oil": 2.79,
            "biodiesel - rapeseed oil": 2.79,
            "synthetic diesel - economic allocation": 3.15,
            "synthetic diesel - energy allocation": 3.15,
            "cng": 2.68,
            "lpg": 3.01,
            "biogas - sewage sludge": 2.68,
            "biogas - biowaste": 2.68,
            "syngas": 2.68,
            "electrolysis": 0,
            "smr - natural gas": 0,
            "smr - natural gas with CCS": 0,
            "smr - biogas": 0,
            "smr - biogas with CCS": 0,
            "atr - natural gas": 0,
            "atr - natural gas with CCS": 0,
            "atr - biogas": 0,
            "atr - biogas with CCS": 0,
            "coal gasification": 0,
            "wood gasification": 0,
            "wood gasification (Swiss forest)": 0,
            "wood gasification with CCS": 0,
            "wood gasification with CCS (Swiss forest)": 0,
            "wood gasification with EF": 0,
            "wood gasification with EF (Swiss forest)": 0,
            "wood gasification with EF with CCS": 0,
            "wood gasification with EF with CCS (Swiss forest)": 0,
        }

        if fuel_blend is None:
            fuel_blend = dict()

        fuel_types = ["diesel", "cng", "hydrogen"]

        for fuel_type in fuel_types:
            primary, secondary, primary_share, secondary_share = self.find_fuel_shares(
                fuel_blend, fuel_type
            )

            fuel_blend[fuel_type] = {
                "primary": {
                    "type": primary,
                    "share": primary_share,
                    "lhv": fuels_lhv[primary],
                    "CO2": fuels_CO2[primary],
                },
                "secondary": {
                    "type": secondary,
                    "share": secondary_share,
                    "lhv": fuels_lhv[secondary],
                    "CO2": fuels_CO2[secondary],
                },
            }

        return fuel_blend
