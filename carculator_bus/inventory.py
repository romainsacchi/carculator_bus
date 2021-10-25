import csv
import glob
import itertools
from inspect import currentframe, getframeinfo
from pathlib import Path

import numpy as np
import xarray as xr
from pypardiso import spsolve
from scipy import sparse

from . import DATA_DIR
from .background_systems import BackgroundSystemModel
from .export import ExportInventory
from .geomap import Geomap
from .utils import build_fleet_array

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

REMIND_FILES_DIR = DATA_DIR / "IAM"


def get_dict_input():
    """
    Load a dictionary with tuple ("name of activity", "location", "unit", "reference product") as key, row/column
    indices as values.

    :return: dictionary with `label:index` pairs.
    :rtype: dict

    """
    filename = "dict_inputs_A_matrix.csv"
    filepath = DATA_DIR / filename
    if not filepath.is_file():
        raise FileNotFoundError("The dictionary of activity labels could not be found.")
    csv_dict = {}
    count = 0
    with open(filepath, encoding="utf-8") as files:
        input_dict = csv.reader(files, delimiter=";")
        for row in input_dict:
            if "(" in row[1]:
                new_str = row[1].replace("(", "")
                new_str = new_str.replace(")", "")
                new_str = [string.strip() for string in new_str.split(",") if string]
                tuple_ = ()
                for string in new_str:

                    if "low population" in string:
                        string = "low population density, long-term"
                        tuple_ += (string,)
                        break

                    if "ground-" in string:
                        if len(new_str) > 2:
                            string = "ground-, long-term"
                            tuple_ += (string,)
                            break

                        string = "ground-"
                        tuple_ += (string,)
                        break

                    tuple_ += (string.replace("'", ""),)
                csv_dict[(row[0], tuple_, row[2])] = count
            else:
                csv_dict[(row[0], row[1], row[2], row[3])] = count
            count += 1

    return csv_dict


class InventoryCalculation:
    """
    Build and solve the inventory for results characterization and inventory export

    Vehicles to be analyzed can be filtered by passing a `scope` dictionary.
    Some assumptions in the background system can also be adjusted by passing a `background_configuration` dictionary.

    .. code-block:: python

        scope = {
                        'powertrain':['BEV', 'FCEV', 'ICEV-p'],
                    }
        bc = {'country':'CH', # considers electricity network losses for Switzerland
              'custom electricity mix' : [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                          [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                          [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                          [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                         ], # in this case, 100% nuclear for the second year
              'fuel blend':{
                  'cng':{ #specify fuel bland for compressed gas
                        'primary fuel':{
                            'type':'biogas - sewage sludge',
                            'share':[0.9, 0.8, 0.7, 0.6] # shares per year. Must total 1 for each year.
                            },
                        'secondary fuel':{
                            'type':'syngas',
                            'share': [0.1, 0.2, 0.3, 0.4]
                            }
                        },
                 'diesel':{
                        'primary fuel':{
                            'type':'synthetic diesel - energy allocation',
                            'share':[0.9, 0.8, 0.7, 0.6]
                            },
                        'secondary fuel':{
                            'type':'biodiesel - cooking oil',
                            'share': [0.1, 0.2, 0.3, 0.4]
                            }
                        },
                'hydrogen':{
                        'primary fuel':{'type':'electrolysis', 'share':[1, 0, 0, 0]},
                        'secondary fuel':{'type':'smr - natural gas', 'share':[0, 1, 1, 1]}
                        }
                    },
              'energy storage': {
                  'electric': {
                      'origin': 'NO'
                  },
                  'hydrogen': {
                      'type':'carbon fiber'
                  }
              }
             }

        InventoryCalculation(CarModel.array,
                            background_configuration=background_configuration,
                            scope=scope,
                            scenario="RCP26")

    The `custom electricity mix` key in the background_configuration dictionary defines an electricity mix to apply,
    under the form of one or several array(s), depending on the number of years to analyze,
    that should total 1, of which the indices correspond to:

        - [0]: hydro-power
        - [1]: nuclear
        - [2]: natural gas
        - [3]: solar power
        - [4]: wind power
        - [5]: biomass
        - [6]: coal
        - [7]: oil
        - [8]: geothermal
        - [9]: waste incineration
        - [10]: biogas with CCS
        - [11]: biomass with CCS
        - [12]: coal with CCS
        - [13]: natural gas with CCS
        - [14]: wood with CCS

    If none is given, the electricity mix corresponding to the country specified in `country` will be selected.
    If no country is specified, Europe applies.

    The `primary` and `secondary` fuel keys contain an array with shares of alternative fuel for each year, to create a custom blend.
    If none is provided, a blend provided by the Integrated Assessment model REMIND is used, which will depend on the REMIND energy scenario selected.

    Here is a list of available fuel pathways:


    Hydrogen technologies
    --------------------
    "electrolysis"
    "smr - natural gas"
    "smr - natural gas with CCS"
    "smr - biogas"
    "smr - biogas with CCS"
    "coal gasification"
    "wood gasification"
    "wood gasification with CCS"
    "wood gasification with EF"
    "wood gasification with EF with CCS"
    "atr - natural gas"
    "atr - natural gas with CCS"
    "atr - biogas"
    "atr - biogas with CCS"

    Natural gas technologies
    ------------------------
    cng
    biogas - sewage sludge
    biogas - biowaste
    syngas

    Diesel technologies
    -------------------
    diesel
    biodiesel - algae
    biodiesel - cooking oil
    synthetic diesel - economic allocation
    synthetic diesel - energy allocation


    :ivar array: array from the CarModel class
    :vartype array: CarModel.array
    :ivar fuel_blend: a dictionary that contains fuel blend shares per type of fuel.
    :ivar scope: dictionary that contains filters for narrowing the analysis
    :ivar background_configuration: dictionary that contains choices for background system
    :ivar scenario: REMIND energy scenario to use ("SSP2-Baseline": business-as-usual,
                                                    "SSP2-PkBudg1100": limits cumulative GHG emissions to 1,100 gigatons by 2100,
                                                    "static": no forward-looking modification of the background inventories).
                    "SSP2-Baseline" selected by default.


    .. code-block:: python

    """

    def __init__(
        self,
        bm,
        scope=None,
        background_configuration=None,
        scenario="SSP2-Base",
        method="recipe",
        method_type="midpoint",
    ):

        if scope is None:
            scope = {
                "size": bm.array.coords["size"].values.tolist(),
                "powertrain": bm.array.coords["powertrain"].values.tolist(),
                "year": bm.array.coords["year"].values.tolist(),
                "fu": {"unit": "pkm", "quantity": 1},
            }
        else:
            scope["size"] = scope.get("size", bm.array.coords["size"].values.tolist())
            scope["powertrain"] = scope.get(
                "powertrain", bm.array.coords["powertrain"].values.tolist()
            )
            scope["year"] = scope.get("year", bm.array.coords["year"].values.tolist())
            scope["fu"] = scope.get("fu", {"unit": "pkm", "quantity": 1})

        if "unit" not in scope["fu"]:
            scope["fu"]["unit"] = "pkm"
        else:
            if scope["fu"]["unit"] not in ["pkm", "vkm"]:
                raise NameError(
                    "Incorrect specification of functional unit. Must be 'pkm' or 'vkm'."
                )

        if "quantity" not in scope["fu"]:
            scope["fu"]["quantity"] = 1
        else:
            try:
                float(scope["fu"]["quantity"])
            except Exception as err:
                raise ValueError(
                    "Incorrect quantity for the functional unit defined."
                ) from err

        self.scope = scope
        self.scenario = scenario
        self.geo = Geomap()

        # Check if a fleet composition is specified
        if "fleet" in self.scope["fu"]:

            if isinstance(self.scope["fu"]["fleet"], xr.DataArray):
                self.fleet = self.scope["fu"]["fleet"]
            else:

                # check if a path as string is provided
                if isinstance(self.scope["fu"]["fleet"], str):
                    filepath = Path(self.scope["fu"]["fleet"])

                # check if instance of pathlib is provided instead
                elif isinstance(self.scope["fu"]["fleet"], Path):
                    filepath = self.scope["fu"]["fleet"]

                else:
                    raise TypeError(
                        "The format used to specify fleet compositions is not valid."
                        "a_matrix file path that points to a CSV file is expected. "
                        "Or an array of type xarray.DataArray."
                    )

                if not filepath.is_file():
                    raise FileNotFoundError(
                        "The CSV file that contains fleet composition could not be found."
                    )

                if filepath.suffix != ".csv":
                    raise TypeError(
                        "a_matrix CSV file is expected to build the fleet composition."
                    )

                self.fleet = build_fleet_array(filepath, self.scope)

        else:
            self.fleet = None

        array = bm.array.sel(
            powertrain=self.scope["powertrain"],
            year=self.scope["year"],
            size=self.scope["size"],
        )

        # store some important specs for inventory documentation
        self.specs = bm.array.sel(
            parameter=[
                "combustion power",
                "electric power",
                "combustion power share",
                "lifetime kilometers",
                "kilometers per year",
                "daily distance",
                "operation time",
                "TtW efficiency",
                "TtW energy",
                "fuel cell system efficiency",
                "electric energy stored",
                "oxidation energy stored",
                "energy battery mass",
                "initial passengers capacity",
                "average passengers",
                "curb mass",
                "driving mass",
            ]
        )

        self.compliant_vehicles = (
            array.sel(parameter="is_available")
            # * array.sel(parameter="is_compliant")
            * np.where(array.sel(parameter="has_schedule_issue") == 0, 1, 0)
            * np.where(array.sel(parameter="is_too_heavy") == 0, 1, 0)
            * 1
        )

        self.array = array.stack(desired=["size", "powertrain", "year"])

        self.iterations = len(bm.array.value.values)

        self.number_of_cars = (
            len(self.scope["size"])
            * len(self.scope["powertrain"])
            * len(self.scope["year"])
        )

        self.array_inputs = {
            x: i for i, x in enumerate(list(self.array.parameter.values), 0)
        }
        self.array_powertrains = {
            x: i for i, x in enumerate(list(self.array.powertrain.values), 0)
        }

        self.background_configuration = background_configuration or {}

        if "energy storage" not in self.background_configuration:
            self.background_configuration["energy storage"] = {
                "electric": {"origin": "CN"}
            }
            self.background_configuration["energy storage"]["electric"].update(
                bm.energy_storage["electric"]
            )

        if "electric" not in self.background_configuration["energy storage"]:
            self.background_configuration["energy storage"]["electric"] = {
                "origin": "CN",
                "BEV-opp": "LTO",
                "BEV-depot": "NMC-111",
                "BEV-motion": "LTO",
                "FCEV": "NMC-111",
                "HEV-d": "NMC-111",
            }
        else:
            if (
                "origin"
                not in self.background_configuration["energy storage"]["electric"]
            ):
                self.background_configuration["energy storage"]["electric"][
                    "origin"
                ] = "CN"

        self.inputs = get_dict_input()
        self.background_system = BackgroundSystemModel()
        self.country = (
            self.background_configuration["country"]
            if "country" in self.background_configuration
            else bm.country
        )
        self.add_additional_activities()
        self.rev_inputs = self.get_rev_dict_input()
        self.a_matrix = self.get_a_matrix()
        self.mix = self.define_electricity_mix_for_fuel_prep()

        self.fuel_blends = bm.fuel_blend
        self.fuel_dictionary = self.create_fuel_dictionary()
        self.create_fuel_markets()

        if "direct air capture" in self.background_configuration:
            if "heat source" in self.background_configuration["direct air capture"]:
                heat_source = self.background_configuration["direct air capture"][
                    "heat source"
                ]

                if heat_source != "waste heat":
                    self.select_heat_supplier(heat_source)

        self.index_cng = [val for key, val in self.inputs.items() if "ICEV-g" in key[0]]

        self.index_fuel_cell = [
            val for key, val in self.inputs.items() if "FCEV" in key[0]
        ]

        self.map_non_fuel_emissions = {
            (
                "1-Pentene",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "1-Pentene direct emissions, rural",
            (
                "1-Pentene",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "1-Pentene direct emissions, suburban",
            (
                "1-Pentene",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "1-Pentene direct emissions, urban",
            (
                "Acetaldehyde",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Acetaldehyde direct emissions, rural",
            (
                "Acetaldehyde",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Acetaldehyde direct emissions, suburban",
            (
                "Acetaldehyde",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Acetaldehyde direct emissions, urban",
            (
                "Acetone",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Acetone direct emissions, rural",
            (
                "Acetone",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Acetone direct emissions, suburban",
            (
                "Acetone",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Acetone direct emissions, urban",
            (
                "Acrolein",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Acrolein direct emissions, rural",
            (
                "Acrolein",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Acrolein direct emissions, suburban",
            (
                "Acrolein",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Acrolein direct emissions, urban",
            (
                "Ammonia",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Ammonia direct emissions, rural",
            (
                "Ammonia",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Ammonia direct emissions, suburban",
            (
                "Ammonia",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Ammonia direct emissions, urban",
            (
                "Arsenic",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Arsenic direct emissions, rural",
            (
                "Arsenic",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Arsenic direct emissions, suburban",
            (
                "Arsenic",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Arsenic direct emissions, urban",
            (
                "Benzaldehyde",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Benzaldehyde direct emissions, rural",
            (
                "Benzaldehyde",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Benzaldehyde direct emissions, suburban",
            (
                "Benzaldehyde",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Benzaldehyde direct emissions, urban",
            (
                "Benzene",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Benzene direct emissions, rural",
            (
                "Benzene",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Benzene direct emissions, suburban",
            (
                "Benzene",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Benzene direct emissions, urban",
            (
                "Butane",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Butane direct emissions, rural",
            (
                "Butane",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Butane direct emissions, suburban",
            (
                "Butane",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Butane direct emissions, urban",
            (
                "Cadmium",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Cadmium direct emissions, rural",
            (
                "Cadmium",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Cadmium direct emissions, suburban",
            (
                "Cadmium",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Cadmium direct emissions, urban",
            (
                "Carbon monoxide, fossil",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Carbon monoxide direct emissions, rural",
            (
                "Carbon monoxide, fossil",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Carbon monoxide direct emissions, suburban",
            (
                "Carbon monoxide, fossil",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Carbon monoxide direct emissions, urban",
            (
                "Chromium",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Chromium direct emissions, rural",
            (
                "Chromium",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Chromium direct emissions, suburban",
            (
                "Chromium",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Chromium direct emissions, urban",
            (
                "Chromium VI",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Chromium VI direct emissions, rural",
            (
                "Chromium VI",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Chromium VI direct emissions, suburban",
            (
                "Chromium VI",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Chromium VI direct emissions, urban",
            (
                "Copper",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Copper direct emissions, rural",
            (
                "Copper",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Copper direct emissions, suburban",
            (
                "Copper",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Copper direct emissions, urban",
            (
                "Cyclohexane",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Cyclohexane direct emissions, rural",
            (
                "Cyclohexane",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Cyclohexane direct emissions, suburban",
            (
                "Cyclohexane",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Cyclohexane direct emissions, urban",
            (
                "Dinitrogen monoxide",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Dinitrogen oxide direct emissions, rural",
            (
                "Dinitrogen monoxide",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Dinitrogen oxide direct emissions, suburban",
            (
                "Dinitrogen monoxide",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Dinitrogen oxide direct emissions, urban",
            (
                "Ethane",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Ethane direct emissions, rural",
            (
                "Ethane",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Ethane direct emissions, suburban",
            (
                "Ethane",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Ethane direct emissions, urban",
            (
                "Ethene",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Ethene direct emissions, rural",
            (
                "Ethene",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Ethene direct emissions, suburban",
            (
                "Ethene",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Ethene direct emissions, urban",
            (
                "Formaldehyde",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Formaldehyde direct emissions, rural",
            (
                "Formaldehyde",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Formaldehyde direct emissions, suburban",
            (
                "Formaldehyde",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Formaldehyde direct emissions, urban",
            (
                "Heptane",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Heptane direct emissions, rural",
            (
                "Heptane",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Heptane direct emissions, suburban",
            (
                "Heptane",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Heptane direct emissions, urban",
            (
                "Hexane",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Hexane direct emissions, rural",
            (
                "Hexane",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Hexane direct emissions, suburban",
            (
                "Hexane",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Hexane direct emissions, urban",
            (
                "Hydrocarbons, chlorinated",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Hydrocarbons direct emissions, rural",
            (
                "Hydrocarbons, chlorinated",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Hydrocarbons direct emissions, suburban",
            (
                "Hydrocarbons, chlorinated",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Hydrocarbons direct emissions, urban",
            (
                "Mercury",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Mercury direct emissions, rural",
            (
                "Mercury",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Mercury direct emissions, suburban",
            (
                "Mercury",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Mercury direct emissions, urban",
            (
                "Methane, fossil",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Methane direct emissions, rural",
            (
                "Methane, fossil",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Methane direct emissions, suburban",
            (
                "Methane, fossil",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Methane direct emissions, urban",
            (
                "Methyl ethyl ketone",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Methyl ethyl ketone direct emissions, rural",
            (
                "Methyl ethyl ketone",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Methyl ethyl ketone direct emissions, suburban",
            (
                "Methyl ethyl ketone",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Methyl ethyl ketone direct emissions, urban",
            (
                "m-Xylene",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "m-Xylene direct emissions, rural",
            (
                "m-Xylene",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "m-Xylene direct emissions, suburban",
            (
                "m-Xylene",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "m-Xylene direct emissions, urban",
            (
                "Nickel",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Nickel direct emissions, rural",
            (
                "Nickel",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Nickel direct emissions, suburban",
            (
                "Nickel",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Nickel direct emissions, urban",
            (
                "Nitrogen oxides",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Nitrogen oxides direct emissions, rural",
            (
                "Nitrogen oxides",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Nitrogen oxides direct emissions, suburban",
            (
                "Nitrogen oxides",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Nitrogen oxides direct emissions, urban",
            (
                "NMVOC, non-methane volatile organic compounds, unspecified origin",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "NMVOC direct emissions, rural",
            (
                "NMVOC, non-methane volatile organic compounds, unspecified origin",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "NMVOC direct emissions, suburban",
            (
                "NMVOC, non-methane volatile organic compounds, unspecified origin",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "NMVOC direct emissions, urban",
            (
                "o-Xylene",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "o-Xylene direct emissions, rural",
            (
                "o-Xylene",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "o-Xylene direct emissions, suburban",
            (
                "o-Xylene",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "o-Xylene direct emissions, urban",
            (
                "PAH, polycyclic aromatic hydrocarbons",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "PAH, polycyclic aromatic hydrocarbons direct emissions, rural",
            (
                "PAH, polycyclic aromatic hydrocarbons",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "PAH, polycyclic aromatic hydrocarbons direct emissions, suburban",
            (
                "PAH, polycyclic aromatic hydrocarbons",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "PAH, polycyclic aromatic hydrocarbons direct emissions, urban",
            (
                "Particulates, < 2.5 um",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Particulate matters direct emissions, rural",
            (
                "Particulates, < 2.5 um",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Particulate matters direct emissions, suburban",
            (
                "Particulates, < 2.5 um",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Particulate matters direct emissions, urban",
            (
                "Pentane",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Pentane direct emissions, rural",
            (
                "Pentane",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Pentane direct emissions, suburban",
            (
                "Pentane",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Pentane direct emissions, urban",
            (
                "Propane",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Propane direct emissions, rural",
            (
                "Propane",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Propane direct emissions, suburban",
            (
                "Propane",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Propane direct emissions, urban",
            (
                "Propene",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Propene direct emissions, rural",
            (
                "Propene",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Propene direct emissions, suburban",
            (
                "Propene",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Propene direct emissions, urban",
            (
                "Selenium",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Selenium direct emissions, rural",
            (
                "Selenium",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Selenium direct emissions, suburban",
            (
                "Selenium",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Selenium direct emissions, urban",
            (
                "Styrene",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Styrene direct emissions, rural",
            (
                "Styrene",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Styrene direct emissions, suburban",
            (
                "Styrene",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Styrene direct emissions, urban",
            (
                "Toluene",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Toluene direct emissions, rural",
            (
                "Toluene",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Toluene direct emissions, suburban",
            (
                "Toluene",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Toluene direct emissions, urban",
            (
                "Zinc",
                ("air", "low population density, long-term"),
                "kilogram",
            ): "Zinc direct emissions, rural",
            (
                "Zinc",
                ("air", "non-urban air or from high stacks"),
                "kilogram",
            ): "Zinc direct emissions, suburban",
            (
                "Zinc",
                ("air", "urban air close to ground"),
                "kilogram",
            ): "Zinc direct emissions, urban",
        }

        self.index_emissions = [self.inputs[i] for i in self.map_non_fuel_emissions]

        self.map_noise_emissions = {
            (
                "noise, octave 1, day time, urban",
                ("octave 1", "day time", "urban"),
                "joule",
            ): "noise, octave 1, day time, urban",
            (
                "noise, octave 2, day time, urban",
                ("octave 2", "day time", "urban"),
                "joule",
            ): "noise, octave 2, day time, urban",
            (
                "noise, octave 3, day time, urban",
                ("octave 3", "day time", "urban"),
                "joule",
            ): "noise, octave 3, day time, urban",
            (
                "noise, octave 4, day time, urban",
                ("octave 4", "day time", "urban"),
                "joule",
            ): "noise, octave 4, day time, urban",
            (
                "noise, octave 5, day time, urban",
                ("octave 5", "day time", "urban"),
                "joule",
            ): "noise, octave 5, day time, urban",
            (
                "noise, octave 6, day time, urban",
                ("octave 6", "day time", "urban"),
                "joule",
            ): "noise, octave 6, day time, urban",
            (
                "noise, octave 7, day time, urban",
                ("octave 7", "day time", "urban"),
                "joule",
            ): "noise, octave 7, day time, urban",
            (
                "noise, octave 8, day time, urban",
                ("octave 8", "day time", "urban"),
                "joule",
            ): "noise, octave 8, day time, urban",
            (
                "noise, octave 1, day time, suburban",
                ("octave 1", "day time", "suburban"),
                "joule",
            ): "noise, octave 1, day time, suburban",
            (
                "noise, octave 2, day time, suburban",
                ("octave 2", "day time", "suburban"),
                "joule",
            ): "noise, octave 2, day time, suburban",
            (
                "noise, octave 3, day time, suburban",
                ("octave 3", "day time", "suburban"),
                "joule",
            ): "noise, octave 3, day time, suburban",
            (
                "noise, octave 4, day time, suburban",
                ("octave 4", "day time", "suburban"),
                "joule",
            ): "noise, octave 4, day time, suburban",
            (
                "noise, octave 5, day time, suburban",
                ("octave 5", "day time", "suburban"),
                "joule",
            ): "noise, octave 5, day time, suburban",
            (
                "noise, octave 6, day time, suburban",
                ("octave 6", "day time", "suburban"),
                "joule",
            ): "noise, octave 6, day time, suburban",
            (
                "noise, octave 7, day time, suburban",
                ("octave 7", "day time", "suburban"),
                "joule",
            ): "noise, octave 7, day time, suburban",
            (
                "noise, octave 8, day time, suburban",
                ("octave 8", "day time", "suburban"),
                "joule",
            ): "noise, octave 8, day time, suburban",
            (
                "noise, octave 1, day time, rural",
                ("octave 1", "day time", "rural"),
                "joule",
            ): "noise, octave 1, day time, rural",
            (
                "noise, octave 2, day time, rural",
                ("octave 2", "day time", "rural"),
                "joule",
            ): "noise, octave 2, day time, rural",
            (
                "noise, octave 3, day time, rural",
                ("octave 3", "day time", "rural"),
                "joule",
            ): "noise, octave 3, day time, rural",
            (
                "noise, octave 4, day time, rural",
                ("octave 4", "day time", "rural"),
                "joule",
            ): "noise, octave 4, day time, rural",
            (
                "noise, octave 5, day time, rural",
                ("octave 5", "day time", "rural"),
                "joule",
            ): "noise, octave 5, day time, rural",
            (
                "noise, octave 6, day time, rural",
                ("octave 6", "day time", "rural"),
                "joule",
            ): "noise, octave 6, day time, rural",
            (
                "noise, octave 7, day time, rural",
                ("octave 7", "day time", "rural"),
                "joule",
            ): "noise, octave 7, day time, rural",
            (
                "noise, octave 8, day time, rural",
                ("octave 8", "day time", "rural"),
                "joule",
            ): "noise, octave 8, day time, rural",
        }

        self.elec_map = {
            "Hydro": (
                "electricity production, hydro, run-of-river",
                "DE",
                "kilowatt hour",
                "electricity, high voltage",
            ),
            "Nuclear": (
                "electricity production, nuclear, pressure water reactor",
                "DE",
                "kilowatt hour",
                "electricity, high voltage",
            ),
            "Gas": (
                "electricity production, natural gas, conventional power plant",
                "DE",
                "kilowatt hour",
                "electricity, high voltage",
            ),
            "Solar": (
                "electricity production, photovoltaic, 3kWp slanted-roof installation, multi-Si, panel, mounted",
                "DE",
                "kilowatt hour",
                "electricity, low voltage",
            ),
            "Wind": (
                "electricity production, wind, 1-3MW turbine, onshore",
                "DE",
                "kilowatt hour",
                "electricity, high voltage",
            ),
            "Biomass": (
                "heat and power co-generation, wood chips, 6667 kW, state-of-the-art 2014",
                "DE",
                "kilowatt hour",
                "electricity, high voltage",
            ),
            "Coal": (
                "electricity production, hard coal",
                "DE",
                "kilowatt hour",
                "electricity, high voltage",
            ),
            "Oil": (
                "electricity production, oil",
                "DE",
                "kilowatt hour",
                "electricity, high voltage",
            ),
            "Geo": (
                "electricity production, deep geothermal",
                "DE",
                "kilowatt hour",
                "electricity, high voltage",
            ),
            "Waste": (
                "treatment of municipal solid waste, incineration",
                "DE",
                "kilowatt hour",
                "electricity, for reuse in municipal waste incineration only",
            ),
            "Biogas CCS": (
                "electricity production, at power plant/biogas, post, pipeline 200km, storage 1000m",
                "RER",
                "kilowatt hour",
                "electricity, high voltage",
            ),
            "Biomass CCS": (
                "electricity production, at BIGCC power plant 450MW, pre, pipeline 200km, storage 1000m",
                "RER",
                "kilowatt hour",
                "electricity, high voltage",
            ),
            "Coal CCS": (
                "electricity production, at power plant/hard coal, post, pipeline 200km, storage 1000m",
                "RER",
                "kilowatt hour",
                "electricity, high voltage",
            ),
            "Gas CCS": (
                "electricity production, at power plant/natural gas, post, pipeline 200km, storage 1000m",
                "RER",
                "kilowatt hour",
                "electricity, high voltage",
            ),
            "Wood CCS": (
                "electricity production, at wood burning power plant 20 MW, truck 25km, post, pipeline 200km, storage 1000m",
                "RER",
                "kilowatt hour",
                "electricity, high voltage",
            ),
        }

        self.index_noise = [self.inputs[i] for i in self.map_noise_emissions]

        self.list_cat, self.split_indices = self.get_split_indices()

        self.method = method

        if self.method == "recipe":
            self.method_type = method_type
        else:
            self.method_type = "midpoint"

        self.impact_categories = self.get_dict_impact_categories()

        # Load the b_matrix matrix
        self.b_matrix = None

    def find_inputs_indices(
        self,
        must_contain: list = None,
        must_also_contain: list = None,
        excludes: list = None,
    ) -> list:
        """
        Return the position(s) of certain keys of `self.inputs` in `a_matrix`.
        :return:
        """
        must_contain = must_contain or []
        must_also_contain = must_also_contain or []
        excludes = excludes or []

        if len(must_also_contain) > 0:
            if len(excludes) > 0:
                return [
                    val
                    for key, val in self.inputs.items()
                    if all(i in key[0] for i in must_contain)
                    and any(i in key[0] for i in must_also_contain)
                    and not any(i in key[0] for i in excludes)
                ]

            return [
                val
                for key, val in self.inputs.items()
                if all(i in key[0] for i in must_contain)
                and any(i in key[0] for i in must_also_contain)
            ]
        else:
            return [
                val
                for key, val in self.inputs.items()
                if all(i in key[0] for i in must_contain)
            ]

    def get_results_table(self, split, sensitivity=False):
        """
        Format an xarray.DataArray array to receive the results.

        :param sensitivity:
        :param split: "components" or "impact categories". Split by impact categories only applicable when "endpoint" level is applied.
        :return: xarrray.DataArray
        """

        if split == "components":
            cat = [
                "direct - exhaust",
                "direct - non-exhaust",
                "energy chain",
                "maintenance",
                "glider",
                "powertrain",
                "energy storage",
                "road",
                "EoL",
            ]

        dict_impact_cat = list(self.impact_categories.keys())

        if not sensitivity:

            response = xr.DataArray(
                np.zeros(
                    (
                        self.b_matrix.shape[1],
                        len(self.scope["size"]),
                        len(self.scope["powertrain"]),
                        len(self.scope["year"]),
                        len(cat),
                        self.iterations,
                    )
                ),
                coords=[
                    dict_impact_cat,
                    self.scope["size"],
                    self.scope["powertrain"],
                    self.scope["year"],
                    cat,
                    np.arange(0, self.iterations),
                ],
                dims=[
                    "impact_category",
                    "size",
                    "powertrain",
                    "year",
                    "impact",
                    "value",
                ],
            )

        else:
            params = self.array.value.values.tolist()
            response = xr.DataArray(
                np.zeros(
                    (
                        self.b_matrix.shape[1],
                        len(self.scope["size"]),
                        len(self.scope["powertrain"]),
                        len(self.scope["year"]),
                        self.iterations,
                    )
                ),
                coords=[
                    dict_impact_cat,
                    self.scope["size"],
                    self.scope["powertrain"],
                    self.scope["year"],
                    params,
                ],
                dims=["impact_category", "size", "powertrain", "year", "parameter"],
            )

        return response

    def get_sulfur_content(self, location, fuel, year):
        """
        Return the sulfur content in the fuel.
        If a region is passed, the average sulfur content over
        the countries the region contains is returned.
        :param location: str. a_matrix country or region ISO code
        :param fuel: str. "diesel" or "gasoline"
        :return: float. Sulfur content in ppm.
        """

        try:
            int(year)
        except Exception as err:
            raise ValueError(
                "The year for which to fetch sulfur concentration values is not valid."
            ) from err

        if location in self.background_system.sulfur.country.values:
            sulfur_concentration = (
                self.background_system.sulfur.sel(
                    country=location, year=year, fuel=fuel
                )
                .sum()
                .values
            )
        else:
            # If the geography is in fact a region,
            # we need to calculate hte average sulfur content
            # across the region

            list_countries = self.geo.iam_to_ecoinvent_location(location)
            list_countries = [
                c
                for c in list_countries
                if c in self.background_system.sulfur.country.values
            ]

            if len(list_countries) > 0:

                sulfur_concentration = (
                    self.background_system.sulfur.sel(
                        country=list_countries,
                        year=year,
                        fuel=fuel,
                    )
                    .mean()
                    .values
                )
            else:

                # if we do not have the sulfur concentration for the required country, we pick Europe
                print(
                    f"The sulfur content for {fuel} fuel in {location} could not be found. European average sulfur content is used instead."
                )
                sulfur_concentration = (
                    self.background_system.sulfur.sel(
                        country="RER", year=year, fuel=fuel
                    )
                    .sum()
                    .values
                )
        return sulfur_concentration

    def get_split_indices(self):
        """
        Return list of indices to split the results into categories.

        :return: list of indices
        :rtype: list
        """
        filename = "dict_split.csv"
        filepath = DATA_DIR / filename
        if not filepath.is_file():
            raise FileNotFoundError("The dictionary of splits could not be found.")

        with open(filepath, encoding="utf-8") as file:
            csv_list = [[val.strip() for val in r.split(";")] for r in file.readlines()]

        (_, _, *header), *data = csv_list

        csv_dict = {}
        for row in data:
            key, sub_key, *values = row

            if key in csv_dict:
                if sub_key in csv_dict[key]:
                    csv_dict[key][sub_key].append(
                        {"search by": values[0], "search for": values[1]}
                    )
                else:
                    csv_dict[key][sub_key] = [
                        {"search by": values[0], "search for": values[1]}
                    ]
            else:
                csv_dict[key] = {
                    sub_key: [{"search by": values[0], "search for": values[1]}]
                }

        flatten = itertools.chain.from_iterable

        direct_emissions = {}
        list_emissions = []

        direct_emissions["direct - exhaust"] = []
        direct_emissions["direct - exhaust"].append(
            self.inputs[("Carbon dioxide, fossil", ("air",), "kilogram")]
        )
        direct_emissions["direct - exhaust"].append(
            self.inputs[
                ("Carbon dioxide, from soil or biomass stock", ("air",), "kilogram")
            ]
        )
        direct_emissions["direct - exhaust"].append(
            self.inputs[("Methane, fossil", ("air",), "kilogram")]
        )

        # emissions from the diesel generator of the trolleybus
        direct_emissions["direct - exhaust"].append(
            self.inputs[
                (
                    "diesel, burned in diesel-electric generating set, 18.5kW",
                    "GLO",
                    "megajoule",
                    "diesel, burned in diesel-electric generating set, 18.5kW",
                )
            ]
        )

        direct_emissions["direct - exhaust"].extend(self.index_emissions)
        direct_emissions["direct - exhaust"].extend(self.index_noise)

        list_emissions.append(direct_emissions["direct - exhaust"])

        for cat in csv_dict["components"]:
            direct_emissions[cat] = list(
                flatten(
                    [
                        self.get_index_of_flows([l["search for"]], l["search by"])
                        for l in csv_dict["components"][cat]
                    ]
                )
            )
            list_emissions.append(direct_emissions[cat])

        list_ind = list(direct_emissions.values())
        max_length = max(map(len, list_ind))
        for row in list_ind:
            while len(row) < max_length:
                row.extend([len(self.inputs) - 1])
        return list(direct_emissions.keys()), list_ind

    def calculate_impacts(self, split="components", sensitivity=False):

        self.b_matrix = self.get_b_matrix()
        # Prepare an array to store the results
        results = self.get_results_table(split, sensitivity=sensitivity)

        # Create electricity and fuel market datasets
        self.create_electricity_market_for_fuel_prep()

        # Create electricity market dataset for battery production
        self.create_electricity_market_for_battery_production()

        # Add rows for fleet vehicles, if any
        if isinstance(self.fleet, xr.core.dataarray.DataArray):
            self.build_fleet_vehicles()

            # Update number of cars
            self.number_of_cars += len(self.scope["year"]) * len(
                self.scope["powertrain"]
            )

            # Update b_matrix matrix
            self.b_matrix = self.get_b_matrix()

        # Fill in the a_matrix matrix with car parameters
        self.set_inputs_in_a_matrix(self.array.values)
        self.a_matrix = np.nan_to_num(self.a_matrix)

        new_arr = np.float32(
            np.zeros(
                (
                    self.a_matrix.shape[1],
                    self.b_matrix.shape[1],
                    len(self.scope["year"]),
                )
            )
        )

        demand_vector = np.zeros((np.shape(self.a_matrix)[1]))

        # Collect indices of activities contributing to the first level for year `y`
        ind_trucks = [
            val
            for key, val in self.inputs.items()
            if "transport, passenger bus, " in key[0] and "market" not in key[0]
        ]
        arr = self.a_matrix[0, : -self.number_of_cars, ind_trucks].sum(axis=0)
        ind = np.nonzero(arr)[0]

        if self.scenario != "static":
            b_mat = self.b_matrix.interp(
                year=self.scope["year"], kwargs={"fill_value": "extrapolate"}
            ).values
        else:
            b_mat = self.b_matrix[0].values

        for demand in ind:
            demand_vector[:] = 0
            demand_vector[demand] = 1
            inverse = np.float32(
                spsolve(sparse.csr_matrix(self.a_matrix[0]), demand_vector.T)
            )

            if self.scenario == "static":
                new_arr[demand] = np.float32(inverse * b_mat).sum(axis=-1).T[..., None]
            else:
                new_arr[demand] = np.float32(inverse * b_mat).sum(axis=-1).T

        shape = (
            self.iterations,
            len(self.scope["size"]),
            len(self.scope["powertrain"]),
            len(self.scope["year"]),
            self.a_matrix.shape[1],
        )

        arr = (
            self.a_matrix[:, :, -self.number_of_cars :]
            .transpose(0, 2, 1)
            .reshape(shape)
            * new_arr.transpose(1, 2, 0)[:, None, None, None, ...]
            * -1
        )
        arr = arr[..., self.split_indices].sum(axis=-1)

        if sensitivity:

            results[...] = arr.transpose(0, 2, 3, 4, 5, 1).sum(axis=-2)
            results /= results.sel(parameter="reference")

        else:
            results[...] = arr.transpose(0, 2, 3, 4, 5, 1)

        if self.scope["fu"]["unit"] == "pkm":
            load_factor = 1

        if self.scope["fu"]["unit"] == "vkm":
            load_factor = np.resize(
                self.array[self.array_inputs["average passengers"]].values,
                (
                    1,
                    len(self.scope["size"]),
                    len(self.scope["powertrain"]),
                    len(self.scope["year"]),
                    1,
                    1,
                ),
            )

        if sensitivity:
            return (
                results.astype("float32")
                * load_factor
                * self.compliant_vehicles.values[None, ...]
            )

        return results.astype("float32") * load_factor * self.compliant_vehicles

    def add_additional_activities(self):
        # Add as many rows and columns as cars to consider
        # Also add additional columns and rows for electricity markets
        # for fuel preparation and energy battery production

        maximum = max(self.inputs.values())

        for year in self.scope["year"]:

            if {"ICEV-d", "HEV-d", "PHEV-d"}.intersection(
                set(self.scope["powertrain"])
            ):
                maximum += 1
                self.inputs[
                    (
                        f"fuel supply for diesel vehicles, {year}",
                        self.country,
                        "kilogram",
                        "fuel",
                    )
                ] = maximum

            if {"ICEV-g"}.intersection(set(self.scope["powertrain"])):
                maximum += 1
                self.inputs[
                    (
                        f"fuel supply for gas vehicles, {year}",
                        self.country,
                        "kilogram",
                        "fuel",
                    )
                ] = maximum

            if {"FCEV"}.intersection(set(self.scope["powertrain"])):
                maximum += 1
                self.inputs[
                    (
                        f"fuel supply for hydrogen vehicles, {year}",
                        self.country,
                        "kilogram",
                        "fuel",
                    )
                ] = maximum

            if {"BEV-opp", "BEV-depot", "BEV-motion", "PHEV-d"}.intersection(
                set(self.scope["powertrain"])
            ):
                maximum += 1
                self.inputs[
                    (
                        f"electricity supply for electric vehicles, {year}",
                        self.country,
                        "kilowatt hour",
                        "electricity, low voltage, for battery electric vehicles",
                    )
                ] = maximum

            maximum += 1
            self.inputs[
                (
                    f"electricity market for fuel preparation, {year}",
                    self.country,
                    "kilowatt hour",
                    "electricity, low voltage",
                )
            ] = maximum

            maximum += 1
            self.inputs[
                (
                    f"electricity market for energy storage production, {year}",
                    self.background_configuration["energy storage"]["electric"][
                        "origin"
                    ],
                    "kilowatt hour",
                    "electricity, low voltage, for energy storage production",
                )
            ] = maximum

        for size in self.scope["size"]:
            for powertrain in self.scope["powertrain"]:
                for year in self.scope["year"]:
                    maximum += 1

                    if year < 1992:
                        euro_class = "EURO-0"
                    elif 1992 <= year < 1995:
                        euro_class = "EURO-I"
                    elif 1995 <= year < 1999:
                        euro_class = "EURO-II"
                    elif 1999 <= year < 2005:
                        euro_class = "EURO-III"
                    elif 2005 <= year < 2008:
                        euro_class = "EURO-IV"
                    elif 2008 <= year < 2012:
                        euro_class = "EURO-V"
                    else:
                        euro_class = "EURO-VI"

                    d_map_size = {
                        "9m": "9m midibus",
                        "13m-city": "13m single deck urban bus",
                        "13m-coach": "13m single deck coach bus",
                        "13m-city-double": "13m double deck urban bus",
                        "13m-coach-double": "13m double deck coach bus",
                        "18m": "18m articulated urban bus",
                    }

                    if self.scope["fu"]["unit"] == "pkm":
                        unit = "passenger-kilometer"
                    else:
                        unit = "kilometer"

                    if powertrain in ["BEV-depot", "BEV-opp", "BEV-motion", "FCEV"]:

                        if powertrain == "FCEV":
                            name = f"transport, passenger bus, {powertrain}, {d_map_size[size]}, {year}"
                        else:
                            name = f"transport, passenger bus, {powertrain}, {self.background_configuration['energy storage']['electric'][powertrain]} battery, {d_map_size[size]}, {year}"

                        self.inputs[
                            (name, self.country, unit, "transport, passenger bus")
                        ] = maximum

                    else:
                        name = f"transport, passenger bus, {powertrain}, {d_map_size[size]}, {year}, {euro_class}"

                        self.inputs[
                            (
                                name,
                                self.country,
                                unit,
                                f"transport, passenger bus, {euro_class}",
                            )
                        ] = maximum

    def add_additional_activities_for_export(self):
        # Add as many rows and columns as trucks to consider
        # Also add additional columns and rows for electricity markets
        # for fuel preparation and energy battery production

        maximum = max(self.inputs.values())

        for size in self.scope["size"]:
            for powertrain in self.scope["powertrain"]:
                for year in self.scope["year"]:
                    maximum += 1

                    if year < 1992:
                        euro_class = "EURO-0"
                    elif 1992 <= year < 1995:
                        euro_class = "EURO-I"
                    elif 1995 <= year < 1999:
                        euro_class = "EURO-II"
                    elif 1999 <= year < 2005:
                        euro_class = "EURO-III"
                    elif 2005 <= year < 2008:
                        euro_class = "EURO-IV"
                    elif 2008 <= year < 2012:
                        euro_class = "EURO-V"
                    else:
                        euro_class = "EURO-VI"

                    d_map_size = {
                        "9m": "9m midibus",
                        "13m-city": "13m single deck urban bus",
                        "13m-coach": "13m single deck coach bus",
                        "13m-city-double": "13m double deck urban bus",
                        "13m-coach-double": "13m double deck coach bus",
                        "18m": "18m articulated urban bus",
                    }

                    if powertrain in ["BEV-depot", "BEV-opp", "BEV-motion", "FCEV"]:

                        if powertrain == "FCEV":
                            name = f"Passenger bus, {powertrain}, {d_map_size[size]}, {year}"

                        else:
                            name = f"Passenger bus, {powertrain}, {self.background_configuration['energy storage']['electric'][powertrain]} battery, {d_map_size[size]}, {year}"

                        self.inputs[
                            (
                                name,
                                self.country,
                                "unit",
                                "Passenger bus",
                            )
                        ] = maximum

                    else:
                        name = f"Passenger bus, {powertrain}, {d_map_size[size]}, {year}, {euro_class}"

                        self.inputs[
                            (name, self.country, "unit", f"Passenger bus, {euro_class}")
                        ] = maximum

    def get_a_matrix(self):
        """
        Load the a_matrix matrix. The a_matrix matrix contains exchanges of products (rows) between activities (columns).

        :return: a_matrix matrix with three dimensions of shape (number of values, number of products, number of activities).
        :rtype: numpy.ndarray

        """
        filename = "A_matrix.csv"
        filepath = (
            Path(getframeinfo(currentframe()).filename)
            .resolve()
            .parent.joinpath(f"data/{filename}")
        )
        if not filepath.is_file():
            raise FileNotFoundError("The technology matrix could not be found.")

        # build matrix a_matrix from coordinates
        a_matrix_coords = np.genfromtxt(filepath, delimiter=";")
        indices_i = a_matrix_coords[:, 0].astype(int)
        indices_j = a_matrix_coords[:, 1].astype(int)
        initial_a_matrix = sparse.csr_matrix(
            (a_matrix_coords[:, 2], (indices_i, indices_j))
        ).toarray()

        new_a_matrix = np.identity(len(self.inputs))

        new_a_matrix[
            0 : np.shape(initial_a_matrix)[0], 0 : np.shape(initial_a_matrix)[0]
        ] = initial_a_matrix

        # Resize the matrix to fit the number of iterations in `array`
        new_a_matrix = np.resize(
            new_a_matrix,
            (self.array.shape[1], new_a_matrix.shape[0], new_a_matrix.shape[1]),
        )
        return new_a_matrix

    def build_fleet_vehicles(self):

        # additional cars
        n_cars = (
            len(self.scope["year"]) * len(self.scope["powertrain"])
            + (len(self.scope["year"]) * len(self.scope["size"]))
            + len(self.scope["year"])
        )
        self.a_matrix = np.pad(self.a_matrix, ((0, 0), (0, n_cars), (0, n_cars)))

        maximum = max(self.inputs.values())

        for powertrain in self.scope["powertrain"]:

            for year in self.scope["year"]:

                # share of the powertrain that year, all sizes
                share_pt = (
                    self.fleet.sel(powertrain=powertrain, variable=year).sum().values
                )

                name = f"transport, passenger bus, fleet average, {powertrain}, {year}"

                maximum += 1

                if self.scope["fu"]["unit"] == "pkm":
                    unit = "passenger-kilometer"
                else:
                    unit = "kilometer"

                self.inputs[
                    (
                        name,
                        self.background_configuration["country"],
                        unit,
                        "transport, passenger bus, fleet average",
                    )
                ] = maximum

                self.a_matrix[:, maximum, maximum] = 1

                if share_pt > 0:
                    for size in self.fleet.coords["size"].values:
                        for vin_year in range(min(self.scope["year"]), year + 1):
                            if vin_year in self.fleet.vintage_year:
                                fleet_share = (
                                    self.fleet.sel(
                                        powertrain=powertrain,
                                        vintage_year=vin_year,
                                        size=size,
                                        variable=year,
                                    )
                                    .sum()
                                    .values
                                    / share_pt
                                )

                                if fleet_share > 0:

                                    car_index = [
                                        self.inputs[i]
                                        for i in self.inputs
                                        if all(
                                            item in i[0]
                                            for item in [
                                                powertrain,
                                                str(vin_year),
                                                size,
                                                "transport, ",
                                            ]
                                        )
                                    ][0]
                                    car_inputs = (
                                        self.a_matrix[:, : car_index - 1, car_index]
                                        * fleet_share
                                    )

                                    self.a_matrix[
                                        :, : car_index - 1, maximum
                                    ] += car_inputs

                    # Fuel and electricity supply must be from the fleet year, not the vintage year

                    d_map_fuel = {
                        "ICEV-d": "diesel",
                        "ICEV-g": "gas",
                        "HEV-d": "diesel",
                        "PHEV-d": "diesel",
                        "BEV-depot": "electric",
                        "BEV-opp": "electric",
                        "BEV-motion": "electric",
                        "FCEV": "hydrogen",
                    }

                    ind_supply = [
                        val
                        for key, val in self.inputs.items()
                        if f"supply for {d_map_fuel[powertrain]} vehicles, " in key[0]
                    ]
                    amount_fuel = self.a_matrix[:, ind_supply, maximum].sum(axis=1)

                    # zero out initial fuel inputs
                    self.a_matrix[:, ind_supply, maximum] = 0

                    # set saved amount to current fuel supply provider
                    current_provider = [
                        self.inputs[i]
                        for i in self.inputs
                        if f"supply for {d_map_fuel[powertrain]} vehicles, {year}"
                        in i[0]
                    ]
                    self.a_matrix[:, current_provider, maximum] = amount_fuel

                    if powertrain in ["PHEV-d"]:
                        ind_supply = [
                            self.inputs[i]
                            for i in self.inputs
                            if "supply for electric vehicles, " in i[0]
                        ]
                        amount_fuel = self.a_matrix[:, ind_supply, maximum].sum(axis=1)

                        # zero out initial fuel inputs
                        self.a_matrix[:, ind_supply, maximum] = 0

                        # set saved amount to current fuel supply provider
                        current_provider = [
                            self.inputs[i]
                            for i in self.inputs
                            if f"supply for electric vehicles, {year}" in i[0]
                        ]
                        self.a_matrix[:, current_provider, maximum] = amount_fuel

        # We also want to produce a fleet average vehicle, with all powertrain types, but for a specific size class

        for year in self.scope["year"]:

            for size in self.scope["size"]:

                # share of that year, all powertrains
                share_pt = self.fleet.sel(variable=year, size=size).sum().values

                d_map_size = {
                    "9m": "9m midibus",
                    "13m-city": "13m single deck urban bus",
                    "13m-coach": "13m single deck coach bus",
                    "13m-city-double": "13m double deck urban bus",
                    "13m-coach-double": "13m double deck coach bus",
                    "18m": "18m articulated urban bus",
                }

                name = f"transport, passenger bus, fleet average, {d_map_size[size]}, {year}"
                ref = "transport, passenger bus, fleet average"

                maximum += 1

                if self.scope["fu"]["unit"] == "pkm":
                    unit = "passenger-kilometer"
                else:
                    unit = "kilometer"

                self.inputs[
                    (
                        name,
                        self.background_configuration["country"],
                        unit,
                        ref,
                    )
                ] = maximum

                self.a_matrix[:, maximum, maximum] = 1

                if share_pt > 0:
                    for powertrain in self.fleet.coords["powertrain"].values:
                        for vin_year in range(min(self.scope["year"]), year + 1):
                            if vin_year in self.fleet.vintage_year:

                                fleet_share = (
                                    self.fleet.sel(
                                        powertrain=powertrain,
                                        vintage_year=vin_year,
                                        size=size,
                                        variable=year,
                                    )
                                    .sum()
                                    .values
                                    / share_pt
                                )

                                if fleet_share > 0:
                                    car_index = [
                                        self.inputs[i]
                                        for i in self.inputs
                                        if all(
                                            item in i[0]
                                            for item in [
                                                powertrain,
                                                str(vin_year),
                                                size,
                                                "transport",
                                            ]
                                        )
                                    ][0]

                                    car_inputs = (
                                        self.a_matrix[:, : car_index - 1, car_index]
                                        * fleet_share
                                    )

                                    self.a_matrix[
                                        :, : car_index - 1, maximum
                                    ] += car_inputs

                # Fuel and electricity supply must be from the fleet year, not the vintage year
                d_map_fuel = {
                    "ICEV-d": "diesel",
                    "ICEV-g": "gas",
                    "HEV-d": "diesel",
                    "PHEV-d": "diesel",
                    "BEV-depot": "electric",
                    "BEV-opp": "electric",
                    "BEV-motion": "electric",
                    "FCEV": "hydrogen",
                }

                for fuel_type in set(d_map_fuel.values()):

                    ind_supply = [
                        self.inputs[i]
                        for i in self.inputs
                        if f"supply for {fuel_type} vehicles, " in i[0]
                    ]

                    amount_fuel = self.a_matrix[:, ind_supply, maximum].sum(axis=1)

                    if amount_fuel < 0:
                        # zero out initial fuel inputs
                        self.a_matrix[:, ind_supply, maximum] = 0

                        # set saved amount to current fuel supply provider
                        current_provider = [
                            self.inputs[i]
                            for i in self.inputs
                            if f"supply for {fuel_type} vehicles, {year}" in i[0]
                        ]

                        self.a_matrix[:, current_provider, maximum] = amount_fuel

        # And finally, a size and powertrain fleet average truck
        # share of that year, all powertrains
        share_pt = self.fleet.sel(variable=year).sum().values

        name = f"transport, passenger bus, fleet average, {year}"
        ref = "transport, passenger bus, fleet average"

        maximum += 1

        if self.scope["fu"]["unit"] == "pkm":
            unit = "passenger-kilometer"
        else:
            unit = "kilometer"

        self.inputs[
            (
                name,
                self.background_configuration["country"],
                unit,
                ref,
            )
        ] = maximum

        self.a_matrix[:, maximum, maximum] = 1

        if share_pt > 0:
            for powertrain in self.fleet.coords["powertrain"].values:
                for size in self.fleet.coords["size"].values:
                    for vin_year in range(min(self.scope["year"]), year + 1):
                        if vin_year in self.fleet.vintage_year:

                            fleet_share = (
                                self.fleet.sel(
                                    powertrain=powertrain,
                                    vintage_year=vin_year,
                                    size=size,
                                    variable=year,
                                )
                                .sum()
                                .values
                                / share_pt
                            )

                            if fleet_share > 0:
                                car_index = [
                                    self.inputs[i]
                                    for i in self.inputs
                                    if all(
                                        item in i[0]
                                        for item in [
                                            powertrain,
                                            str(vin_year),
                                            size,
                                            "transport",
                                        ]
                                    )
                                ][0]

                                car_inputs = (
                                    self.a_matrix[:, : car_index - 1, car_index]
                                    * fleet_share
                                )

                                self.a_matrix[:, : car_index - 1, maximum] += car_inputs

        # Fuel and electricity supply must be from the fleet year, not the vintage year
        d_map_fuel = {
            "ICEV-d": "diesel",
            "ICEV-g": "gas",
            "HEV-d": "diesel",
            "PHEV-d": "diesel",
            "BEV-depot": "electric",
            "BEV-opp": "electric",
            "BEV-motion": "electric",
            "FCEV": "hydrogen",
        }

        for fuel_type in set(d_map_fuel.values()):

            ind_supply = [
                val
                for key, val in self.inputs.items()
                if f"supply for {fuel_type} vehicles, " in key[0]
            ]

            amount_fuel = self.a_matrix[:, ind_supply, maximum].sum(axis=1)

            if amount_fuel < 0:
                # zero out initial fuel inputs
                self.a_matrix[:, ind_supply, maximum] = 0

                # set saved amount to current fuel supply provider
                current_provider = [
                    val
                    for key, val in self.inputs.items()
                    if f"supply for {fuel_type} vehicles, {year}" in key[0]
                ]

                self.a_matrix[:, current_provider, maximum] = amount_fuel

    def get_b_matrix(self):
        """
        Load the b_matrix matrix. The b_matrix matrix contains impact assessment figures for a give impact assessment method,
        per unit of activity. Its length column-wise equals the length of the a_matrix matrix row-wise.
        Its length row-wise equals the number of impact assessment methods.

        :return: an array with impact values per unit of activity for each method.
        :rtype: numpy.ndarray

        """

        if self.method == "recipe":
            if self.method_type == "midpoint":
                list_file_names = glob.glob(
                    f"{REMIND_FILES_DIR}/*recipe_midpoint*{self.scenario}*.csv"
                )
                list_file_names = sorted(list_file_names)
                matrix_b = np.zeros((len(list_file_names), 21, len(self.inputs)))
            elif self.method_type == "endpoint":
                list_file_names = glob.glob(
                    f"{REMIND_FILES_DIR}/*recipe_endpoint*{self.scenario}*.csv"
                )
                list_file_names = sorted(list_file_names)
                matrix_b = np.zeros((len(list_file_names), 4, len(self.inputs)))
            else:
                raise TypeError(
                    "The LCIA method type should be either 'midpoint' or 'endpoint'."
                )

        else:
            list_file_names = glob.glob(
                f"{REMIND_FILES_DIR}/*ilcd*{self.scenario}*.csv"
            )
            matrix_b = np.zeros((len(list_file_names), 19, len(self.inputs)))

        for f, filepath in enumerate(list_file_names):
            initial_b = np.genfromtxt(filepath, delimiter=";")

            new_b = np.zeros(
                (
                    np.shape(initial_b)[0],
                    len(self.inputs),
                )
            )

            new_b[0 : np.shape(initial_b)[0], 0 : np.shape(initial_b)[1]] = initial_b

            matrix_b[f] = new_b

        list_impact_categories = list(self.impact_categories.keys())

        if self.scenario != "static":
            response = xr.DataArray(
                matrix_b,
                coords=[
                    [2005, 2010, 2020, 2030, 2040, 2050],
                    list_impact_categories,
                    list(self.inputs.keys()),
                ],
                dims=["year", "category", "activity"],
            )
        else:
            response = xr.DataArray(
                matrix_b,
                coords=[[2020], list_impact_categories, list(self.inputs.keys())],
                dims=["year", "category", "activity"],
            )

        return response

    def get_dict_impact_categories(self):
        """
        Load a dictionary with available impact assessment methods as keys, and assessment level and categories as values.

        ..code-block:: python

            {'recipe': {'midpoint': ['freshwater ecotoxicity',
                                   'human toxicity',
                                   'marine ecotoxicity',
                                   'terrestrial ecotoxicity',
                                   'metal depletion',
                                   'agricultural land occupation',
                                   'climate change',
                                   'fossil depletion',
                                   'freshwater eutrophication',
                                   'ionising radiation',
                                   'marine eutrophication',
                                   'natural land transformation',
                                   'ozone depletion',
                                   'particulate matter formation',
                                   'photochemical oxidant formation',
                                   'terrestrial acidification',
                                   'urban land occupation',
                                   'water depletion',
                                   'human noise',
                                   'primary energy, non-renewable',
                                   'primary energy, renewable']
                       }
           }

        :return: dictionary
        :rtype: dict
        """
        filename = "dict_impact_categories.csv"
        filepath = DATA_DIR / filename
        if not filepath.is_file():
            raise FileNotFoundError(
                "The dictionary of impact categories could not be found."
            )

        csv_dict = {}

        with open(filepath, encoding="utf-8") as f:
            input_dict = csv.reader(f, delimiter=";")
            for row in input_dict:
                if row[0] == self.method and row[3] == self.method_type:
                    csv_dict[row[2]] = {
                        "method": row[1],
                        "category": row[2],
                        "type": row[3],
                        "abbreviation": row[4],
                        "unit": row[5],
                        "source": row[6],
                    }

        return csv_dict

    def get_rev_dict_input(self):
        """
        Reverse the self.inputs dictionary.

        :return: reversed dictionary
        :rtype: dict
        """
        return {v: k for k, v in self.inputs.items()}

    def get_index_vehicle_from_array(
        self, items_to_look_for, items_to_look_for_also=None, method="or"
    ):
        """
        Return list of row/column indices of self.array of labels that contain the string defined in `items_to_look_for`.

        :param items_to_look_for_also:
        :param method:
        :param items_to_look_for: string to search for
        :return: list
        """
        if not isinstance(items_to_look_for, list):
            items_to_look_for = [items_to_look_for]

        if not items_to_look_for_also is None:
            if not isinstance(items_to_look_for_also, list):
                items_to_look_for_also = [items_to_look_for_also]

        list_vehicles = self.array.desired.values

        if method == "or":
            return [
                c
                for c, v in enumerate(list_vehicles)
                if set(items_to_look_for).intersection(v)
            ]

        return [
            c
            for c, v in enumerate(list_vehicles)
            if set(items_to_look_for).intersection(v)
            and set(items_to_look_for_also).intersection(v)
        ]

    def get_index_of_flows(self, items_to_look_for, search_by="name"):
        """
        Return list of row/column indices of self.a_matrix of labels that contain the string defined in `items_to_look_for`.

        :param items_to_look_for: string
        :param search_by: "name" or "compartment" (for elementary flows)
        :return: list of row/column indices
        :rtype: list
        """
        if search_by == "name":
            return [
                int(self.inputs[c])
                for c in self.inputs
                if all(ele in c[0].lower() for ele in items_to_look_for)
            ]

        return [
            int(self.inputs[c])
            for c in self.inputs
            if all(ele in c[1] for ele in items_to_look_for)
        ]

    def resize_a_matrix_for_export(self):

        indices_to_remove = []

        d_map_size = {
            "9m midibus": "9m",
            "13m single deck urban bus": "13m-city",
            "13m single deck coach bus": "13m-coach",
            "13m double deck urban bus": "13m-city-double",
            "13m double deck coach bus": "13m-coach-double",
            "18m articulated urban bus": "18m",
        }

        for ind in self.inputs:
            if (
                "passenger bus, " in ind[0].lower()
                and "fleet average" not in ind[0]
                and "market" not in ind[0]
            ):

                if "transport" in ind[0]:

                    if "BEV" in ind[0]:
                        (_, _, powertrain, _, size, year) = [
                            x.strip() for x in ind[0].split(", ")
                        ]

                    elif "FCEV" in ind[0]:
                        (_, _, powertrain, size, year) = [
                            x.strip() for x in ind[0].split(", ")
                        ]

                    else:
                        (
                            _,
                            _,
                            powertrain,
                            size,
                            year,
                            _,
                        ) = [x.strip() for x in ind[0].split(", ")]
                    size = d_map_size[size]
                else:
                    if "BEV" in ind[0]:
                        _, powertrain, _, size, year = ind[0].split(", ")
                    elif "FCEV" in ind[0]:
                        _, powertrain, size, year = ind[0].split(", ")
                    else:
                        _, powertrain, size, year, _ = ind[0].split(", ")

                    size = d_map_size[size]

                if (
                    self.compliant_vehicles.sel(
                        powertrain=powertrain, size=size, year=int(year)
                    )
                    == 0
                ):
                    indices_to_remove.append(self.inputs[ind])
                    self.rev_inputs.pop(self.inputs[ind])

        indices_to_preserve = [
            i for i in range(self.a_matrix.shape[1]) if i not in indices_to_remove
        ]

        self.a_matrix = self.a_matrix[
            np.ix_(
                range(self.a_matrix.shape[0]), indices_to_preserve, indices_to_preserve
            )
        ]

        self.rev_inputs = dict(enumerate(self.rev_inputs.values()))
        self.inputs = dict(enumerate(self.rev_inputs.values()))

    def export_lci(
        self,
        presamples=True,
        ecoinvent_compatibility=True,
        ecoinvent_version="3.7",
        db_name="carculator db",
        create_vehicle_datasets=True,
    ):
        """
        Export the inventory as a dictionary. Also return a list of arrays that contain pre-sampled random values if
        :param db_name:
        :meth:`stochastic` of :class:`CarModel` class has been called.

        :param presamples: boolean.
        :param ecoinvent_compatibility: bool. If True, compatible with ecoinvent. If False, compatible with REMIND-ecoinvent.
        :param ecoinvent_version: str. "3.5", "3.6" or "uvek"
        :return: inventory, and optionally, list of arrays containing pre-sampled values.
        :rtype: list
        """
        self.inputs = get_dict_input()
        self.background_system = BackgroundSystemModel()
        self.add_additional_activities()
        self.rev_inputs = self.get_rev_dict_input()
        self.a_matrix = self.get_a_matrix()

        if create_vehicle_datasets:

            # add vehicles datasets
            self.add_additional_activities_for_export()

            # Update dictionary
            self.rev_inputs = self.get_rev_dict_input()

            # resize a_matrix matrix
            self.a_matrix = self.get_a_matrix()

            # Create electricity and fuel market datasets
            self.create_electricity_market_for_fuel_prep()

            # Create electricity market dataset for battery production
            self.create_electricity_market_for_battery_production()

            # Create fuel markets
            self.create_fuel_markets()
            self.fuel_dictionary = self.create_fuel_dictionary()

            self.set_inputs_in_a_matrix_for_export(self.array.values)

        else:

            # Create electricity and fuel market datasets
            self.create_electricity_market_for_fuel_prep()

            # Create electricity market dataset for battery production
            self.create_electricity_market_for_battery_production()

            # Create fuel markets
            self.fuel_dictionary = self.create_fuel_dictionary()

            self.set_inputs_in_a_matrix(self.array.values)

        # Add rows for fleet vehicles, if any
        if isinstance(self.fleet, xr.core.dataarray.DataArray):
            print("Building fleet average vehicles...")
            self.build_fleet_vehicles()

            # Update dictionary
            self.rev_inputs = self.get_rev_dict_input()

            # Update number of cars
            self.number_of_cars += len(self.scope["year"]) * len(
                self.scope["powertrain"]
            )

        # if the inventories are meant to link to `premise` databases
        # we need to remove the additional electricity input
        # in the fuel market datasets
        if not ecoinvent_compatibility:
            fuel_markets = [
                val for key, val in self.inputs.items() if "fuel market for" in key[0]
            ]
            electricity_inputs = [
                val
                for key, val in self.inputs.items()
                if "electricity market for" in key[0]
            ]
            self.a_matrix[
                np.ix_(range(self.a_matrix.shape[0]), electricity_inputs, fuel_markets)
            ] = 0

        # Remove vehicles not compliant or available
        self.resize_a_matrix_for_export()

        if presamples:
            lci, array = ExportInventory(
                self.a_matrix, self.rev_inputs, db_name=db_name
            ).write_lci(
                presamples=presamples,
                ecoinvent_compatibility=ecoinvent_compatibility,
                ecoinvent_version=ecoinvent_version,
                vehicle_specs=self.specs,
            )
            return lci, array

        lci = ExportInventory(
            self.a_matrix, self.rev_inputs, db_name=db_name
        ).write_lci(
            presamples=presamples,
            ecoinvent_compatibility=ecoinvent_compatibility,
            ecoinvent_version=ecoinvent_version,
            vehicle_specs=self.specs,
        )
        return lci

    def export_lci_to_bw(
        self,
        presamples=True,
        ecoinvent_compatibility=True,
        ecoinvent_version="3.7",
        db_name="carculator db",
        create_vehicle_datasets=True,
        forbidden_activities=None,
    ):
        """
        Export the inventory as a `brightway2` bw2io.importers.base_lci.LCIImporter object
        with the inventory in the `data` attribute.

        .. code-block:: python

            # get the inventory
            i, _ = ic.export_lci_to_bw()

            # import it in a Brightway2 project
            i.match_database('ecoinvent 3.6 cutoff', fields=('name', 'unit', 'location', 'reference product'))
            i.match_database("biosphere3", fields=('name', 'unit', 'categories'))
            i.match_database(fields=('name', 'unit', 'location', 'reference product'))
            i.match_database(fields=('name', 'unit', 'categories'))

            # Create an additional biosphere database for the few flows that do not
            # exist in "biosphere3"
            i.create_new_biosphere("additional_biosphere", relink=True)

            # Check if all exchanges link
            i.statistics()

            # Register the database
            i.write_database()

        :return: LCIImport object that can be directly registered in a `brightway2` project.
        :rtype: bw2io.importers.base_lci.LCIImporter
        """
        self.inputs = get_dict_input()
        self.background_system = BackgroundSystemModel()
        self.add_additional_activities()
        self.rev_inputs = self.get_rev_dict_input()
        self.a_matrix = self.get_a_matrix()

        if create_vehicle_datasets:

            # add vehicles datasets
            self.add_additional_activities_for_export()

            # Update dictionary
            self.rev_inputs = self.get_rev_dict_input()

            # resize a_matrix matrix
            self.a_matrix = self.get_a_matrix()

            # Create electricity and fuel market datasets
            self.create_electricity_market_for_fuel_prep()

            # Create electricity market dataset for battery production
            self.create_electricity_market_for_battery_production()

            # Create fuel markets
            self.create_fuel_markets()
            self.fuel_dictionary = self.create_fuel_dictionary()

            self.set_inputs_in_a_matrix_for_export(self.array.values)

        else:

            # Create electricity and fuel market datasets
            self.create_electricity_market_for_fuel_prep()

            # Create electricity market dataset for battery production
            self.create_electricity_market_for_battery_production()

            # Create fuel markets
            self.create_fuel_markets()
            self.fuel_dictionary = self.create_fuel_dictionary()

            self.set_inputs_in_a_matrix(self.array.values)

        # Add rows for fleet vehicles, if any
        if isinstance(self.fleet, xr.core.dataarray.DataArray):
            print("Building fleet average vehicles...")
            self.build_fleet_vehicles()

            # Update dictionary
            self.rev_inputs = self.get_rev_dict_input()

            # Update number of cars
            self.number_of_cars += len(self.scope["year"]) * len(
                self.scope["powertrain"]
            )

        # if the inventories are meant to link to `premise` databases
        # we need to remove the additional electricity input
        # in the fuel market datasets
        if not ecoinvent_compatibility:
            fuel_markets = [
                self.inputs[a] for a in self.inputs if "fuel market for" in a[0]
            ]
            electricity_inputs = [
                self.inputs[a] for a in self.inputs if "electricity market for" in a[0]
            ]
            self.a_matrix[
                np.ix_(range(self.a_matrix.shape[0]), electricity_inputs, fuel_markets)
            ] = 0

        # Remove vehicles not compliant or available
        self.resize_a_matrix_for_export()

        if presamples:
            lci, array = ExportInventory(
                self.a_matrix, self.rev_inputs, db_name=db_name
            ).write_lci_to_bw(
                presamples=presamples,
                ecoinvent_compatibility=ecoinvent_compatibility,
                ecoinvent_version=ecoinvent_version,
                forbidden_activities=forbidden_activities,
                vehicle_specs=self.specs,
            )
            return lci, array

        lci = ExportInventory(
            self.a_matrix, self.rev_inputs, db_name=db_name
        ).write_lci_to_bw(
            presamples=presamples,
            ecoinvent_compatibility=ecoinvent_compatibility,
            ecoinvent_version=ecoinvent_version,
            forbidden_activities=forbidden_activities,
            vehicle_specs=self.specs,
        )
        return lci

    def export_lci_to_excel(
        self,
        directory=None,
        ecoinvent_compatibility=True,
        ecoinvent_version="3.7",
        software_compatibility="brightway2",
        filename=None,
        create_vehicle_datasets=True,
        forbidden_activities=None,
        export_format="file",
    ):
        """
        Export the inventory as an Excel file (if the destination software is Brightway2) or a CSV file (if the destination software is Simapro) file.
        Also return the file path where the file is stored.

        :param filename:
        :param directory: directory where to save the file.
        :type directory: str
        :param ecoinvent_compatibility: If True, compatible with ecoinvent. If False, compatible with REMIND-ecoinvent.
        :param ecoinvent_version: "3.6", "3.5" or "uvek"
        :param software_compatibility: "brightway2" or "simapro"
        :return: file path where the file is stored.
        :rtype: str
        """

        if software_compatibility not in ("brightway2", "simapro"):
            raise NameError(
                "The destination software argument is not valid. Choose between 'brightway2' or 'simapro'."
            )

        # Simapro inventory only for ecoinvent 3.6 or UVEK
        if software_compatibility == "simapro":
            if ecoinvent_version not in ("3.6", "uvek"):
                print(
                    "Simapro-compatible inventory export is only available for ecoinvent 3.6 or UVEK."
                )
                return None
            ecoinvent_compatibility = True

        self.inputs = get_dict_input()
        self.background_system = BackgroundSystemModel()
        self.add_additional_activities()
        self.rev_inputs = self.get_rev_dict_input()
        self.a_matrix = self.get_a_matrix()

        if create_vehicle_datasets:

            # add vehicles datasets
            self.add_additional_activities_for_export()

            # Update dictionary
            self.rev_inputs = self.get_rev_dict_input()

            # resize a_matrix matrix
            self.a_matrix = self.get_a_matrix()

            # Create electricity and fuel market datasets
            self.create_electricity_market_for_fuel_prep()

            # Create fuel markets
            self.create_fuel_markets()
            self.fuel_dictionary = self.create_fuel_dictionary()

            # Create electricity market dataset for battery production
            self.create_electricity_market_for_battery_production()

            self.set_inputs_in_a_matrix_for_export(self.array.values)

        else:

            # Create electricity and fuel market datasets
            self.create_electricity_market_for_fuel_prep()

            # Create electricity market dataset for battery production
            self.create_electricity_market_for_battery_production()

            # Create fuel markets
            self.create_fuel_markets()
            self.fuel_dictionary = self.create_fuel_dictionary()

            self.set_inputs_in_a_matrix(self.array.values)

        # Add rows for fleet vehicles, if any
        if isinstance(self.fleet, xr.core.dataarray.DataArray):
            print("Building fleet average vehicles...")
            self.build_fleet_vehicles()

            # Update dictionary
            self.rev_inputs = self.get_rev_dict_input()

            # Update number of cars
            self.number_of_cars += len(self.scope["year"]) * len(
                self.scope["powertrain"]
            )

        # if the inventories are meant to link to `premise` databases
        # we need to remove the additional electricity input
        # in the fuel market datasets
        if not ecoinvent_compatibility:
            fuel_markets = [
                self.inputs[a] for a in self.inputs if "fuel market for" in a[0]
            ]
            electricity_inputs = [
                self.inputs[a] for a in self.inputs if "electricity market for" in a[0]
            ]
            self.a_matrix[
                np.ix_(range(self.a_matrix.shape[0]), electricity_inputs, fuel_markets)
            ] = 0

        # Remove vehicles not compliant or available
        self.resize_a_matrix_for_export()

        filepath = ExportInventory(
            self.a_matrix, self.rev_inputs, db_name=filename or "carculator db"
        ).write_lci_to_excel(
            directory=directory,
            ecoinvent_compatibility=ecoinvent_compatibility,
            ecoinvent_version=ecoinvent_version,
            software_compatibility=software_compatibility,
            filename=filename,
            forbidden_activities=forbidden_activities,
            export_format=export_format,
            vehicle_specs=self.specs,
        )
        return filepath

    def define_electricity_mix_for_fuel_prep(self):
        """
        This function defines a fuel mix based either on user-defined mix, or on default mixes for a given country.
        The mix is calculated as the average mix, weighted by the distribution of annually driven kilometers.
        :return:
        """

        if "custom electricity mix" in self.background_configuration:
            # If a special electricity mix is specified, we use it
            if not np.allclose(
                np.sum(self.background_configuration["custom electricity mix"], axis=1),
                [1] * len(self.scope["year"]),
            ):
                raise ValueError("The custom electricity mixes are not valid")

            mix = self.background_configuration["custom electricity mix"]

            if np.shape(mix)[0] != len(self.scope["year"]):
                raise ValueError(
                    f"The number of electricity mixes ({np.shape(mix)[0]}) must match with the "
                    f"number of years ({len(self.scope['year'])})."
                )

        else:
            use_year = [
                int(i)
                for i in (
                    self.array.values[
                        self.array_inputs["lifetime kilometers"],
                        :,
                        self.get_index_vehicle_from_array(
                            [
                                "BEV-motion",
                                "BEV-depot",
                                "BEV-opp",
                                "FCEV",
                                "PHEV-p",
                                "ICEV-p",
                                "ICEV-d",
                                "HEV-p",
                                "HEV-d",
                                "ICEV-g",
                            ]
                        ),
                    ]
                    / self.array.values[
                        self.array_inputs["kilometers per year"],
                        :,
                        self.get_index_vehicle_from_array(
                            [
                                "BEV-motion",
                                "BEV-depot",
                                "BEV-opp",
                                "FCEV",
                                "PHEV-p",
                                "ICEV-p",
                                "ICEV-d",
                                "HEV-p",
                                "HEV-d",
                                "ICEV-g",
                            ]
                        ),
                    ]
                )
                .mean(axis=1)
                .reshape(-1, len(self.scope["year"]))
                .mean(axis=0)
            ]

            if (
                self.country
                not in self.background_system.electricity_mix.country.values
            ):
                print(
                    f"The electricity mix for {self.country} could not be found. Average European electricity mix is used instead."
                )
                country = "RER"
            else:
                country = self.country

            mix = [
                self.background_system.electricity_mix.sel(
                    country=country,
                    variable=[
                        "Hydro",
                        "Nuclear",
                        "Gas",
                        "Solar",
                        "Wind",
                        "Biomass",
                        "Coal",
                        "Oil",
                        "Geothermal",
                        "Waste",
                        "Biogas CCS",
                        "Biomass CCS",
                        "Coal CCS",
                        "Gas CCS",
                        "Wood CCS",
                    ],
                )
                .interp(
                    year=np.arange(year, year + use_year[y]),
                    kwargs={"fill_value": "extrapolate"},
                )
                .mean(axis=0)
                .values
                if year + use_year[y] <= 2050
                else self.background_system.electricity_mix.sel(
                    country=country,
                    variable=[
                        "Hydro",
                        "Nuclear",
                        "Gas",
                        "Solar",
                        "Wind",
                        "Biomass",
                        "Coal",
                        "Oil",
                        "Geothermal",
                        "Waste",
                        "Biogas CCS",
                        "Biomass CCS",
                        "Coal CCS",
                        "Gas CCS",
                        "Wood CCS",
                    ],
                )
                .interp(
                    year=np.arange(year, 2051), kwargs={"fill_value": "extrapolate"}
                )
                .mean(axis=0)
                .values
                for y, year in enumerate(self.scope["year"])
            ]

        return mix

    def define_renewable_rate_in_mix(self):

        try:
            losses_to_low = float(self.background_system.losses[self.country]["LV"])
        except KeyError:
            # If losses for the country are not found, assume EU average
            losses_to_low = float(self.background_system.losses["RER"]["LV"])

        category_name = (
            "climate change"
            if self.method == "recipe"
            else "climate change - climate change total"
        )

        if self.method_type != "endpoint":
            if self.scenario != "static":
                year = self.scope["year"]
                co2_intensity_tech = (
                    self.b_matrix.sel(
                        category=category_name,
                        activity=list(self.elec_map.values()),
                    )
                    .interp(year=year, kwargs={"fill_value": "extrapolate"})
                    .values
                    * losses_to_low
                ) * 1000
            else:
                year = 2020
                co2_intensity_tech = np.resize(
                    (
                        self.b_matrix.sel(
                            category=category_name,
                            activity=list(self.elec_map.values()),
                            year=year,
                        ).values
                        * losses_to_low
                        * 1000
                    ),
                    (len(self.scope["year"]), 15),
                )
        else:
            co2_intensity_tech = np.zeros((len(self.scope["year"]), 15))

        sum_renew = [
            np.sum([self.mix[x][i] for i in [0, 3, 4, 5, 8]])
            for x in range(0, len(self.mix))
        ]

        return sum_renew, co2_intensity_tech

    def create_electricity_market_for_fuel_prep(self):
        """This function fills the electricity market that supplies battery charging operations
        and hydrogen production through electrolysis.
        """

        try:
            losses_to_low = float(self.background_system.losses[self.country]["LV"])
        except KeyError:
            # If losses for the country are not found, assume EU average
            losses_to_low = float(self.background_system.losses["RER"]["LV"])

        # Fill the electricity markets for battery charging and hydrogen production
        for y, year in enumerate(self.scope["year"]):
            m = np.array(self.mix[y]).reshape(-1, 15, 1)
            col_num = [
                val
                for key, val in self.inputs.items()
                if all(
                    item in key[0]
                    for item in [str(year), "electricity market for fuel preparation"]
                )
            ]
            # Add electricity technology shares
            self.a_matrix[
                np.ix_(
                    np.arange(self.iterations),
                    [self.inputs[val] for val in self.elec_map.values()],
                    col_num,
                )
            ] = (
                m * -1 * losses_to_low
            )

            # Add transmission network for high and medium voltage
            self.a_matrix[
                :,
                self.inputs[
                    (
                        "transmission network construction, electricity, high voltage",
                        "CH",
                        "kilometer",
                        "transmission network, electricity, high voltage",
                    )
                ],
                col_num,
            ] = (
                6.58e-9 * -1 * losses_to_low
            )

            self.a_matrix[
                :,
                self.inputs[
                    (
                        "transmission network construction, electricity, medium voltage",
                        "CH",
                        "kilometer",
                        "transmission network, electricity, medium voltage",
                    )
                ],
                col_num,
            ] = (
                1.86e-8 * -1 * losses_to_low
            )

            self.a_matrix[
                :,
                self.inputs[
                    (
                        "transmission network construction, long-distance",
                        "UCTE",
                        "kilometer",
                        "transmission network, long-distance",
                    )
                ],
                col_num,
            ] = (
                3.17e-10 * -1 * losses_to_low
            )

            # Add distribution network, low voltage
            self.a_matrix[
                :,
                self.inputs[
                    (
                        "distribution network construction, electricity, low voltage",
                        "CH",
                        "kilometer",
                        "distribution network, electricity, low voltage",
                    )
                ],
                col_num,
            ] = (
                8.74e-8 * -1 * losses_to_low
            )

            # Add supply of sulfur hexafluoride for transformers
            self.a_matrix[
                :,
                self.inputs[
                    (
                        "market for sulfur hexafluoride, liquid",
                        "RER",
                        "kilogram",
                        "sulfur hexafluoride, liquid",
                    )
                ],
                col_num,
            ] = (
                (5.4e-8 + 2.99e-9) * -1 * losses_to_low
            )

            # Add SF_6 leakage

            self.a_matrix[
                :, self.inputs[("Sulfur hexafluoride", ("air",), "kilogram")], col_num
            ] = ((5.4e-8 + 2.99e-9) * -1 * losses_to_low)

    def create_electricity_market_for_battery_production(self):
        """
        This funciton fills in the dataset that contains the electricity mix used for manufacturing battery cells
        :return:
        """

        battery_origin = self.background_configuration["energy storage"]["electric"][
            "origin"
        ]

        if battery_origin != "custom electricity mix":

            try:
                losses_to_low = float(
                    self.background_system.losses[battery_origin]["LV"]
                )
            except KeyError:
                losses_to_low = float(self.background_system.losses["CN"]["LV"])

            if (
                battery_origin
                not in self.background_system.electricity_mix.country.values
            ):
                print(
                    "The electricity mix for {} could not be found. Average Chinese electricity mix is used for "
                    "battery manufacture instead.".format(self.country)
                )
                battery_origin = "CN"

            mix_battery_manufacturing = (
                self.background_system.electricity_mix.sel(
                    country=battery_origin,
                    variable=[
                        "Hydro",
                        "Nuclear",
                        "Gas",
                        "Solar",
                        "Wind",
                        "Biomass",
                        "Coal",
                        "Oil",
                        "Geothermal",
                        "Waste",
                        "Biogas CCS",
                        "Biomass CCS",
                        "Coal CCS",
                        "Gas CCS",
                        "Wood CCS",
                    ],
                )
                .interp(year=self.scope["year"], kwargs={"fill_value": "extrapolate"})
                .values
            )

        else:
            # electricity mix for battery manufacturing same as `custom electricity mix`
            mix_battery_manufacturing = self.mix
            losses_to_low = 1.1

        # Fill the electricity markets for battery production
        for y, year in enumerate(self.scope["year"]):
            m = np.array(mix_battery_manufacturing[y]).reshape(-1, 15, 1)

            col_num = [
                val
                for key, val in self.inputs.items()
                if all(
                    item in key[0]
                    for item in [
                        str(year),
                        "electricity market for energy storage production",
                    ]
                )
            ]

            self.a_matrix[
                np.ix_(
                    np.arange(self.iterations),
                    [self.inputs[val] for val in self.elec_map.values()],
                    col_num,
                )
            ] = (
                m * losses_to_low * -1
            )

            # Add transmission network for high and medium voltage
            self.a_matrix[
                :,
                self.inputs[
                    (
                        "transmission network construction, electricity, high voltage",
                        "CH",
                        "kilometer",
                        "transmission network, electricity, high voltage",
                    )
                ],
                col_num,
            ] = (
                6.58e-9 * -1 * losses_to_low
            )

            self.a_matrix[
                :,
                self.inputs[
                    (
                        "transmission network construction, electricity, medium voltage",
                        "CH",
                        "kilometer",
                        "transmission network, electricity, medium voltage",
                    )
                ],
                col_num,
            ] = (
                1.86e-8 * -1 * losses_to_low
            )

            self.a_matrix[
                :,
                self.inputs[
                    (
                        "transmission network construction, long-distance",
                        "UCTE",
                        "kilometer",
                        "transmission network, long-distance",
                    )
                ],
                col_num,
            ] = (
                3.17e-10 * -1 * losses_to_low
            )

            # Add distribution network, low voltage
            self.a_matrix[
                :,
                self.inputs[
                    (
                        "distribution network construction, electricity, low voltage",
                        "CH",
                        "kilometer",
                        "distribution network, electricity, low voltage",
                    )
                ],
                col_num,
            ] = (
                8.74e-8 * -1 * losses_to_low
            )

            # Add supply of sulfur hexafluoride for transformers
            self.a_matrix[
                :,
                self.inputs[
                    (
                        "market for sulfur hexafluoride, liquid",
                        "RER",
                        "kilogram",
                        "sulfur hexafluoride, liquid",
                    )
                ],
                col_num,
            ] = (
                (5.4e-8 + 2.99e-9) * -1 * losses_to_low
            )

            # Add SF_6 leakage

            self.a_matrix[
                :, self.inputs[("Sulfur hexafluoride", ("air",), "kilogram")], col_num
            ] = ((5.4e-8 + 2.99e-9) * -1 * losses_to_low)

    def create_fuel_markets(self):
        """
        This function creates markets for fuel, considering a given blend, a given fuel type and a given year
        :return:
        """

        d_dataset_name = {
            "diesel": "fuel supply for diesel vehicles, ",
            "cng": "fuel supply for gas vehicles, ",
            "hydrogen": "fuel supply for hydrogen vehicles, ",
            "electricity": "electricity supply for electric vehicles, ",
        }

        d_map_fuel_pt = {
            "diesel": ["ICEV-d", "HEV-d", "PHEV-d"],
            "cng": ["ICEV-g"],
            "hydrogen": ["FCEV"],
        }

        for fuel_type in self.fuel_blends:

            if any(
                {x}.intersection(set(self.scope["powertrain"]))
                for x in d_map_fuel_pt[fuel_type]
            ):

                primary = self.fuel_blends[fuel_type]["primary"]["type"]
                secondary = self.fuel_blends[fuel_type]["secondary"]["type"]
                primary_share = self.fuel_blends[fuel_type]["primary"]["share"]
                secondary_share = self.fuel_blends[fuel_type]["secondary"]["share"]

                for y, year in enumerate(self.scope["year"]):

                    dataset_name = d_dataset_name[fuel_type] + str(year)
                    fuel_market_index = [
                        val
                        for key, val in self.inputs.items()
                        if key[0] == dataset_name
                    ][0]

                    try:
                        primary_fuel_activity_index = self.inputs[
                            self.fuel_dictionary[primary]["name"]
                        ]
                        secondary_fuel_activity_index = self.inputs[
                            self.fuel_dictionary[secondary]["name"]
                        ]
                    except Exception as err:
                        raise KeyError(
                            f"One of the primary or secondary fuels specified in the fuel blend for {fuel_type} is not valid."
                        ) from err

                    if ~np.isclose(primary_share[y] + secondary_share[y], 1, rtol=1e-3):
                        sum_blend = primary_share[y] + secondary_share[y]
                        raise ValueError(
                            f"The fuel blend for {fuel_type} in {year} is not equal to 1, but {sum_blend}."
                        )

                    self.a_matrix[:, primary_fuel_activity_index, fuel_market_index] = (
                        -1 * primary_share[y]
                    )
                    self.a_matrix[
                        :, secondary_fuel_activity_index, fuel_market_index
                    ] = (-1 * secondary_share[y])

                    def learning_rate_fuel(fuel, year, share, val):
                        if fuel == "electrolysis":
                            # apply some learning rate for electrolysis
                            electrolysis = -0.3538 * (float(year) - 2010) + 58.589
                            electricity = (val - 58 + electrolysis) * share

                        elif fuel == "synthetic diesel - energy allocation":
                            # apply some learning rate for electrolysis
                            hydrogen = 0.42
                            electrolysis = -0.3538 * (float(year) - 2010) + 58.589
                            electricity = val - (hydrogen * 58)
                            electricity += electrolysis * hydrogen
                            electricity *= share

                        elif fuel == "synthetic diesel - economic allocation":
                            # apply some learning rate for electrolysis
                            hydrogen = 0.183
                            electrolysis = -0.3538 * (float(year) - 2010) + 58.589
                            electricity = val - (hydrogen * 58)
                            electricity += electrolysis * hydrogen
                            electricity *= share

                        else:
                            electricity = val * share
                        return electricity

                    additional_electricity_primary = learning_rate_fuel(
                        primary,
                        year,
                        primary_share[y],
                        self.fuel_dictionary[primary]["additional electricity"],
                    )

                    additional_electricity_secondary = learning_rate_fuel(
                        secondary,
                        year,
                        secondary_share[y],
                        self.fuel_dictionary[secondary]["additional electricity"],
                    )

                    additional_electricity = (
                        additional_electricity_primary
                        + additional_electricity_secondary
                    )

                    if additional_electricity > 0:
                        electricity_mix_index = [
                            val
                            for key, val in self.inputs.items()
                            if key[0]
                            == f"electricity market for fuel preparation, {year}"
                        ][0]
                        self.a_matrix[:, electricity_mix_index, fuel_market_index] = (
                            -1 * additional_electricity
                        )

        if any(
            powertrain in self.scope["powertrain"]
            for powertrain in ["BEV-depot", "BEV-opp", "BEV-motion", "PHEV-d"]
        ):
            for year in self.scope["year"]:
                fuel_type = "electricity"
                dataset_name = d_dataset_name[fuel_type] + str(year)
                electricity_market_index = [
                    val for key, val in self.inputs.items() if key[0] == dataset_name
                ][0]
                electricity_mix_index = [
                    val
                    for key, val in self.inputs.items()
                    if key[0] == f"electricity market for fuel preparation, {year}"
                ][0]
                self.a_matrix[:, electricity_mix_index, electricity_market_index] = -1

    def find_inputs(
        self, value_in, value_out, find_input_by="name", zero_out_input=False
    ):
        """
        Finds the exchange inputs to a specified functional unit
        :param find_input_by: can be 'name' or 'unit'
        :param value_in: value to look for
        :param value_out: functional unit output
        :return: indices of all inputs to FU, indices of inputs of intereste
        :rtype: tuple
        """

        if isinstance(value_out, str):
            value_out = [value_out]

        index_output = [
            val
            for value in value_out
            for key, val in self.inputs.items()
            if value.lower() in key[0].lower()
        ]

        demand_vector = np.zeros((np.shape(self.a_matrix)[1]))

        demand_vector[index_output] = 1

        matrix = np.float32(
            spsolve(sparse.csr_matrix(self.a_matrix[0]), demand_vector.T)
        )

        ind_inputs = np.nonzero(matrix)[0]

        if find_input_by == "name":
            ins = [
                i
                for i in ind_inputs
                if value_in.lower() in self.rev_inputs[i][0].lower()
            ]

        else:
            ins = [
                i
                for i in ind_inputs
                if value_in.lower() in self.rev_inputs[i][2].lower()
            ]

        outs = [i for i in ind_inputs if i not in ins]

        sum_supplied = matrix[ins].sum()

        if not zero_out_input:
            return sum_supplied

        # zero out initial inputs
        self.a_matrix[np.ix_(np.arange(0, self.a_matrix.shape[0]), ins, outs)] *= 0
        return None

    def create_fuel_dictionary(self):

        d_fuels = {
            "electrolysis": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from electrolysis, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "smr - natural gas": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from SMR of NG, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "smr - natural gas with CCS": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from SMR of NG, with CCS, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "smr - biogas": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from SMR of biogas, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "smr - biogas with CCS": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from SMR of biogas with CCS, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "coal gasification": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from coal gasification, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "wood gasification": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from heatpipe reformer gasification of woody biomass, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "wood gasification with CCS": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from heatpipe reformer gasification of woody biomass with CCS, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "wood gasification with EF": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from gasification of woody biomass in entrained flow gasifier, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "wood gasification with EF with CCS": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from gasification of woody biomass in entrained flow gasifier, with CCS, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "wood gasification (Swiss forest)": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from heatpipe reformer gasification of woody biomass, at fuelling station",
                    "CH",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "wood gasification with CCS (Swiss forest)": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from heatpipe reformer gasification of woody biomass with CCS, at fuelling station",
                    "CH",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "wood gasification with EF (Swiss forest)": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from gasification of woody biomass in entrained flow gasifier, at fuelling station",
                    "CH",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "wood gasification with EF with CCS (Swiss forest)": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from gasification of woody biomass in entrained flow gasifier, with CCS, at fuelling station",
                    "CH",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "atr - natural gas": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, ATR of NG, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "atr - natural gas with CCS": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, ATR of NG, with CCS, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "atr - biogas": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from ATR of biogas, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "atr - biogas with CCS": {
                "name": (
                    "Hydrogen, gaseous, 700 bar, from ATR of biogas with CCS, at fuelling station",
                    "RER",
                    "kilogram",
                    "Hydrogen, gaseous, 700 bar",
                )
            },
            "cng": {
                "name": (
                    "market for natural gas, high pressure, vehicle grade",
                    "GLO",
                    "kilogram",
                    "natural gas, high pressure, vehicle grade",
                )
            },
            "biogas - sewage sludge": {
                "name": (
                    "Biomethane, gaseous, 5 bar, from sewage sludge fermentation, at fuelling station",
                    "RER",
                    "kilogram",
                    "biomethane, high pressure",
                )
            },
            "biogas - biowaste": {
                "name": (
                    "biomethane from biogas upgrading - biowaste - amine scrubbing",
                    "CH",
                    "kilogram",
                    "biomethane",
                )
            },
            "syngas": {
                "name": (
                    "Methane, synthetic, gaseous, 5 bar, from electrochemical methanation, at fuelling station",
                    "RER",
                    "kilogram",
                    "methane, high pressure",
                )
            },
            "diesel": {
                "name": (
                    "market group for diesel, low-sulfur",
                    "RER",
                    "kilogram",
                    "diesel, low-sulfur",
                )
            },
            "biodiesel - algae": {
                "name": (
                    "Biodiesel, from algae, at fuelling station",
                    "RER",
                    "kilogram",
                    "biodiesel, vehicle grade",
                )
            },
            "biodiesel - cooking oil": {
                "name": (
                    "Biodiesel, from used cooking oil, at fuelling station",
                    "RER",
                    "kilogram",
                    "biodiesel, vehicle grade",
                )
            },
            "biodiesel - rapeseed oil": {
                "name": (
                    "Biodiesel, from rapeseed oil, at fuelling station",
                    "RER",
                    "kilogram",
                    "biodiesel, vehicle grade",
                )
            },
            "biodiesel - palm oil": {
                "name": (
                    "Biodiesel, from palm oil, at fuelling station",
                    "RER",
                    "kilogram",
                    "biodiesel, vehicle grade",
                )
            },
            "synthetic diesel - economic allocation": {
                "name": (
                    "diesel production, synthetic, from electrolysis-based hydrogen, economic allocation, at fuelling station",
                    "RER",
                    "kilogram",
                    "diesel, synthetic, vehicle grade",
                )
            },
            "synthetic diesel - energy allocation": {
                "name": (
                    "diesel production, synthetic, from electrolysis-based hydrogen, energy allocation, at fuelling station",
                    "RER",
                    "kilogram",
                    "diesel, synthetic, vehicle grade",
                )
            },
        }

        for key, val in d_fuels.items():
            if any(
                i in val["name"][0].lower()
                for i in ("synthetic", "hydrogen", "ethanol", "biodiesel")
            ):
                val["additional electricity"] = self.find_inputs(
                    "kilowatt hour", d_fuels[key]["name"][0], "unit"
                )
            else:
                val["additional electricity"] = 0

        for key, val in d_fuels.items():
            if any(
                i in val["name"][0].lower() for i in ("synthetic", "hydrogen", "bio")
            ):
                self.find_inputs(
                    "kilowatt hour", val["name"][0], "unit", zero_out_input=True
                )

        return d_fuels

    def set_inputs_in_a_matrix(self, array):
        """
        Fill-in the a_matrix matrix. Does not return anything. Modifies in place.
        Shape of the a_matrix matrix (values, products, activities).

        :param array: :attr:`array` from :class:`BusModel` class
        """

        # Glider/Frame
        self.a_matrix[
            :,
            self.inputs[
                (
                    "frame, blanks and saddle, for lorry",
                    "RER",
                    "kilogram",
                    "frame, blanks and saddle, for lorry",
                )
            ],
            -self.number_of_cars :,
        ] = (
            (array[self.array_inputs["glider base mass"], :])
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"]]
            * -1
        )

        # Suspension + Brakes
        self.a_matrix[
            :,
            self.inputs[
                ("suspension, for lorry", "RER", "kilogram", "suspension, for lorry")
            ],
            -self.number_of_cars :,
        ] = (
            (
                array[
                    [
                        self.array_inputs["suspension mass"],
                        self.array_inputs["braking system mass"],
                    ],
                    :,
                ].sum(axis=0)
            )
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"]]
            * -1
        )

        # Wheels and tires
        self.a_matrix[
            :,
            self.inputs[
                (
                    "tires and wheels, for lorry",
                    "RER",
                    "kilogram",
                    "tires and wheels, for lorry",
                )
            ],
            -self.number_of_cars :,
        ] = (
            (array[self.array_inputs["wheels and tires mass"], :])
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"]]
            * -1
        )

        # Exhaust
        self.a_matrix[
            :,
            self.inputs[
                (
                    "exhaust system, for lorry",
                    "RER",
                    "kilogram",
                    "exhaust system, for lorry",
                )
            ],
            -self.number_of_cars :,
        ] = (
            (array[self.array_inputs["exhaust system mass"], :])
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"]]
            * -1
        )

        # Electrical system
        self.a_matrix[
            :,
            self.inputs[
                (
                    "power electronics, for lorry",
                    "RER",
                    "kilogram",
                    "power electronics, for lorry",
                )
            ],
            -self.number_of_cars :,
        ] = (
            (array[self.array_inputs["electrical system mass"], :])
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"]]
            * -1
        )

        # Transmission (52% transmission shaft, 36% gearbox + 12% retarder)
        self.a_matrix[
            :,
            self.inputs[
                (
                    "transmission, for lorry",
                    "RER",
                    "kilogram",
                    "transmission, for lorry",
                )
            ],
            -self.number_of_cars :,
        ] = (
            (array[self.array_inputs["transmission mass"], :] * 0.52)
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"]]
            * -1
        )
        self.a_matrix[
            :,
            self.inputs[
                ("gearbox, for lorry", "RER", "kilogram", "gearbox, for lorry")
            ],
            -self.number_of_cars :,
        ] = (
            (array[self.array_inputs["transmission mass"], :] * 0.36)
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"]]
            * -1
        )
        self.a_matrix[
            :,
            self.inputs[
                ("retarder, for lorry", "RER", "kilogram", "retarder, for lorry")
            ],
            -self.number_of_cars :,
        ] = (
            (array[self.array_inputs["transmission mass"], :] * 0.12)
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"]]
            * -1
        )

        # Other components, for non-electric and hybrid trucks
        index = self.get_index_vehicle_from_array(["ICEV-d", "HEV-d", "ICEV-g"])
        ind_a = self.find_inputs_indices(
            must_contain=[
                "transport, passenger bus, ",
            ],
            must_also_contain=[
                "ICEV-d",
                "HEV-d",
                "ICEV-g",
            ],
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "other components, for hybrid electric lorry",
                    "RER",
                    "kilogram",
                    "other components, for hybrid electric lorry",
                )
            ],
            ind_a,
        ] = (
            array[self.array_inputs["other components mass"], :, index]
            / array[self.array_inputs["lifetime kilometers"], :, index]
            / array[self.array_inputs["average passengers"], :, index]
            * -1
        ).T

        # Other components, for electric trucks
        index = self.get_index_vehicle_from_array(
            [
                "BEV-opp",
                "BEV-depot",
                "BEV-motion",
                "FCEV",
            ]
        )
        ind_a = self.find_inputs_indices(
            must_contain=[
                "transport, passenger bus, ",
            ],
            must_also_contain=[
                "BEV",
                "FCEV",
            ],
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "other components, for electric lorry",
                    "RER",
                    "kilogram",
                    "other components, for electric lorry",
                )
            ],
            ind_a,
        ] = (
            array[self.array_inputs["other components mass"], :, index]
            / array[self.array_inputs["lifetime kilometers"], :, index]
            / array[self.array_inputs["average passengers"], :, index]
            * -1
        ).T

        self.a_matrix[
            :,
            self.inputs[
                ("Glider lightweighting", "GLO", "kilogram", "Glider lightweighting")
            ],
            -self.number_of_cars :,
        ] = (
            (
                array[self.array_inputs["lightweighting"]]
                * array[self.array_inputs["glider base mass"]]
            )
            / array[self.array_inputs["lifetime kilometers"]]
            / array[self.array_inputs["average passengers"]]
            * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "maintenance, bus",
                    "CH",
                    "unit",
                    "maintenance, bus",
                )
            ],
            -self.number_of_cars :,
        ] = (
            (
                1
                / array[self.array_inputs["lifetime kilometers"]]
                / array[self.array_inputs["average passengers"]]
            )
            * (array[self.array_inputs["gross mass"]] / 19000)
            * -1
        )

        # Powertrain components

        self.a_matrix[
            :,
            self.inputs[
                (
                    "market for converter, for electric passenger car",
                    "GLO",
                    "kilogram",
                    "converter, for electric passenger car",
                )
            ],
            -self.number_of_cars :,
        ] = (
            array[self.array_inputs["converter mass"], :]
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"], :]
            * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "market for electric motor, electric passenger car",
                    "GLO",
                    "kilogram",
                    "electric motor, electric passenger car",
                )
            ],
            -self.number_of_cars :,
        ] = (
            array[self.array_inputs["electric engine mass"], :]
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"], :]
            * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "market for inverter, for electric passenger car",
                    "GLO",
                    "kilogram",
                    "inverter, for electric passenger car",
                )
            ],
            -self.number_of_cars :,
        ] = (
            array[self.array_inputs["inverter mass"], :]
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"], :]
            * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "market for power distribution unit, for electric passenger car",
                    "GLO",
                    "kilogram",
                    "power distribution unit, for electric passenger car",
                )
            ],
            -self.number_of_cars :,
        ] = (
            array[self.array_inputs["power distribution unit mass"], :]
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"], :]
            * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "internal combustion engine, for lorry",
                    "RER",
                    "kilogram",
                    "internal combustion engine, for lorry",
                )
            ],
            -self.number_of_cars :,
        ] = (
            (
                array[
                    [self.array_inputs[l] for l in ["combustion engine mass"]],
                    :,
                ].sum(axis=0)
            )
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"], :]
            * -1
        )

        self.a_matrix[
            :,
            self.inputs[("Ancillary BoP", "GLO", "kilogram", "Ancillary BoP")],
            -self.number_of_cars :,
        ] = (
            array[self.array_inputs["fuel cell ancillary BoP mass"], :]
            * (1 + array[self.array_inputs["fuel cell lifetime replacements"]])
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"], :]
            * -1
        )

        self.a_matrix[
            :,
            self.inputs[("Essential BoP", "GLO", "kilogram", "Essential BoP")],
            -self.number_of_cars :,
        ] = (
            array[self.array_inputs["fuel cell essential BoP mass"], :]
            * (1 + array[self.array_inputs["fuel cell lifetime replacements"]])
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"], :]
            * -1
        )

        self.a_matrix[
            :,
            self.inputs[("Stack", "GLO", "kilowatt", "Stack")],
            -self.number_of_cars :,
        ] = (
            array[self.array_inputs["fuel cell stack mass"], :]
            * (1 + array[self.array_inputs["fuel cell lifetime replacements"]])
            / 1.02
            / array[self.array_inputs["lifetime kilometers"], :]
            / array[self.array_inputs["average passengers"], :]
            * -1
        )

        # Start of printout

        print(
            "****************** IMPORTANT BACKGROUND PARAMETERS ******************",
            end="\n * ",
        )

        # Energy storage for electric buses

        print(
            f"The country of use is {self.country}.",
            end="\n * ",
        )

        # Battery BoP for all electric and hybrid buses

        index = self.get_index_vehicle_from_array(
            ["BEV-depot", "BEV-opp", "BEV-motion", "FCEV", "HEV-d"]
        )

        ind_a = self.find_inputs_indices(
            must_contain=[
                "transport, passenger bus, ",
            ],
            must_also_contain=["BEV-depot", "BEV-opp", "BEV-motion", "FCEV", "HEV-d"],
        )

        self.a_matrix[
            :, self.inputs[("Battery BoP", "GLO", "kilogram", "Battery BoP")], ind_a
        ] = (
            (
                array[self.array_inputs["battery BoP mass"], :, index]
                * (
                    1
                    + array[
                        self.array_inputs["battery lifetime replacements"], :, index
                    ]
                )
            )
            / array[self.array_inputs["lifetime kilometers"], :, index]
            / array[self.array_inputs["average passengers"], :, index]
            * -1
        ).T

        # Zero out electricity requirement for battery cell manufacture
        for battery_tech in ["NMC-111", "NCA", "LFP", "LTO"]:
            battery_cell_label = (
                f"Battery cell, {battery_tech}",
                "GLO",
                "kilogram",
                "Battery cell",
            )

            # Set an input of electricity, given the country of manufacture
            self.a_matrix[
                :,
                self.inputs[
                    (
                        "market group for electricity, medium voltage",
                        "World",
                        "kilowatt hour",
                        "electricity, medium voltage",
                    )
                ],
                self.inputs[battery_cell_label],
            ] = 0

        # We look into `background_configuration` to see what battery chemistry to
        # use for each bus
        for veh in self.background_configuration["energy storage"]["electric"]:
            if veh != "origin" and veh in self.scope["powertrain"]:
                battery_tech = self.background_configuration["energy storage"][
                    "electric"
                ][veh]
                battery_cell_label = (
                    f"Battery cell, {battery_tech}",
                    "GLO",
                    "kilogram",
                    "Battery cell",
                )

                idx = self.get_index_vehicle_from_array([veh])

                ind_a = self.find_inputs_indices(
                    must_contain=[
                        "transport, passenger bus, ",
                    ],
                    must_also_contain=[veh],
                )

                self.a_matrix[:, self.inputs[battery_cell_label], ind_a] = (
                    (
                        array[self.array_inputs["battery cell mass"], :, idx]
                        * (
                            1
                            + array[
                                self.array_inputs["battery lifetime replacements"],
                                :,
                                idx,
                            ]
                        )
                    )
                    / array[self.array_inputs["lifetime kilometers"], :, idx]
                    / array[self.array_inputs["average passengers"], :, idx]
                    * -1
                ).T

                for year in self.scope["year"]:
                    idx = self.get_index_vehicle_from_array(year, veh, method="and")

                    self.a_matrix[
                        np.ix_(
                            np.arange(self.iterations),
                            self.find_inputs_indices(
                                must_contain=[
                                    "electricity market for energy storage production",
                                ],
                            ),
                            self.find_inputs_indices(
                                must_contain=[
                                    "transport, passenger bus, ",
                                    veh,
                                    str(year),
                                ],
                            ),
                        )
                    ] = (
                        array[
                            self.array_inputs["battery cell production electricity"],
                            :,
                            idx,
                        ].T
                        * self.a_matrix[
                            :,
                            self.inputs[battery_cell_label],
                            self.find_inputs_indices(
                                must_contain=[
                                    "transport, passenger bus, ",
                                    str(year),
                                    veh,
                                ],
                            ),
                        ]
                    ).reshape(
                        self.iterations, 1, -1
                    )

        # Use the inventory of Wolff et al. 2020 for lead acid battery for non-electric and non-hybrid trucks

        ind_a = self.find_inputs_indices(
            must_contain=[
                "transport, passenger bus, ",
            ],
            must_also_contain=["ICEV-d", "ICEV-g"],
        )

        index = self.get_index_vehicle_from_array(["ICEV-d", "ICEV-g"])

        self.a_matrix[
            :,
            self.inputs[
                (
                    "lead acid battery, for lorry",
                    "RER",
                    "kilogram",
                    "lead acid battery, for lorry",
                )
            ],
            ind_a,
        ] = (
            (
                array[self.array_inputs["battery BoP mass"], :, index]
                * (
                    1
                    + array[
                        self.array_inputs["battery lifetime replacements"], :, index
                    ]
                )
            )
            / array[self.array_inputs["lifetime kilometers"], :, index]
            / array[self.array_inputs["average passengers"], :, index]
            * -1
        ).T

        self.a_matrix[
            :,
            self.inputs[
                (
                    "lead acid battery, for lorry",
                    "RER",
                    "kilogram",
                    "lead acid battery, for lorry",
                )
            ],
            ind_a,
        ] += (
            (
                array[self.array_inputs["battery cell mass"], :, index]
                * (
                    1
                    + array[
                        self.array_inputs["battery lifetime replacements"], :, index
                    ]
                )
            )
            / array[self.array_inputs["lifetime kilometers"], :, index]
            / array[self.array_inputs["average passengers"], :, index]
            * -1
        ).T

        # Fuel tank for diesel trucks

        ind_a = self.find_inputs_indices(
            must_contain=[
                "transport, passenger bus, ",
            ],
            must_also_contain=["ICEV-d", "HEV-d"],
        )
        index = self.get_index_vehicle_from_array(["ICEV-d", "HEV-d"])

        self.a_matrix[
            :,
            self.inputs[
                ("fuel tank, for diesel vehicle", "RER", "kilogram", "fuel tank")
            ],
            ind_a,
        ] = (
            array[self.array_inputs["fuel tank mass"], :, index]
            / array[self.array_inputs["lifetime kilometers"], :, index]
            / array[self.array_inputs["average passengers"], :, index]
            * -1
        ).T

        index = self.get_index_vehicle_from_array("ICEV-g")
        self.a_matrix[
            :,
            self.inputs[
                (
                    "Fuel tank, compressed hydrogen gas, 700bar, with aluminium liner",
                    "RER",
                    "kilogram",
                    "Hydrogen tank",
                )
            ],
            self.index_cng,
        ] = (
            array[self.array_inputs["fuel tank mass"], :, index]
            / array[self.array_inputs["lifetime kilometers"], :, index]
            / array[self.array_inputs["average passengers"], :, index]
            * -1
        ).T

        if "hydrogen" in self.background_configuration["energy storage"]:
            # If a customization dict is passed
            hydro_tank_technology = self.background_configuration["energy storage"][
                "hydrogen"
            ]["type"]
        else:
            hydro_tank_technology = "carbon fiber"

        dict_tank_map = {
            "carbon fiber": (
                "Fuel tank, compressed hydrogen gas, 700bar",
                "GLO",
                "kilogram",
                "Fuel tank, compressed hydrogen gas, 700bar",
            ),
            "hdpe": (
                "Fuel tank, compressed hydrogen gas, 700bar, with HDPE liner",
                "RER",
                "kilogram",
                "Hydrogen tank",
            ),
            "aluminium": (
                "Fuel tank, compressed hydrogen gas, 700bar, with aluminium liner",
                "RER",
                "kilogram",
                "Hydrogen tank",
            ),
        }

        index = self.get_index_vehicle_from_array("FCEV")
        self.a_matrix[
            :,
            self.inputs[dict_tank_map[hydro_tank_technology]],
            self.index_fuel_cell,
        ] = (
            array[self.array_inputs["fuel tank mass"], :, index]
            / array[self.array_inputs["lifetime kilometers"], :, index]
            / array[self.array_inputs["average passengers"], :, index]
            * -1
        ).T

        try:
            sum_renew, co2_intensity_tech = self.define_renewable_rate_in_mix()
        except:
            sum_renew = [0] * len(self.scope["year"])
            co2_intensity_tech = [0] * len(self.scope["year"])

        for iyear, year in enumerate(self.scope["year"]):

            if iyear + 1 == len(self.scope["year"]):
                end_str = "\n * "
            else:
                end_str = "\n \t * "

            renew_rate = np.round(sum_renew[iyear] * 100, 0)
            ghg_rate = int(np.sum(co2_intensity_tech[iyear] * self.mix[iyear]))
            print(
                f"in {year}, % of renewable: {renew_rate}, GHG intensity per kWh: {ghg_rate} g. CO2-eq.",
                end=end_str,
            )

            if any(
                True
                for x in ["BEV-opp", "BEV-depot", "BEV-motion", "PHEV-d"]
                if x in self.scope["powertrain"]
            ):

                index = self.get_index_vehicle_from_array(
                    ["BEV-opp", "BEV-depot", "BEV-motion"], year, method="and"
                )

                self.a_matrix[
                    np.ix_(
                        np.arange(self.iterations),
                        self.find_inputs_indices(
                            must_contain=[
                                "electricity supply for electric vehicles",
                                str(year),
                            ]
                        ),
                        self.find_inputs_indices(
                            must_contain=["transport, passenger bus, ", str(year)],
                            must_also_contain=[
                                "BEV-opp",
                                "BEV-depot",
                                "BEV-motion",
                            ],
                        ),
                    )
                ] = (
                    array[self.array_inputs["electricity consumption"], :, index]
                    / array[self.array_inputs["average passengers"], :, index]
                    * -1
                ).T.reshape(
                    self.iterations, 1, -1
                )

        # add the diesel consumption from the generator for BEV-motion buses
        # anterior to 2020
        if any(True for x in ["BEV-motion"] if x in self.scope["powertrain"]):

            index = self.get_index_vehicle_from_array(
                ["BEV-motion"], [2000, 2010], method="and"
            )

            self.a_matrix[
                np.ix_(
                    np.arange(self.iterations),
                    self.find_inputs_indices(
                        must_contain=[
                            "diesel, burned in diesel-electric generating set, 18.5kW",
                        ],
                    ),
                    self.find_inputs_indices(
                        must_contain=["transport, passenger bus, ", "BEV-motion"],
                        must_also_contain=["2000", "2010"],
                    ),
                )
            ] = (
                array[self.array_inputs["oxidation energy stored"], :, index]  # kWh
                / array[self.array_inputs["daily distance"], :, index]  # km
                / array[self.array_inputs["average passengers"], :, index]  # passengers
                * 3.6  # MJ/kWh
                * -1
            ).T.reshape(
                self.iterations, 1, -1
            )

        if "FCEV" in self.scope["powertrain"]:

            index = self.get_index_vehicle_from_array("FCEV")

            print(
                f"{self.fuel_blends['hydrogen']['primary']['type']} is completed by {self.fuel_blends['hydrogen']['secondary']['type']}.",
                end="\n \t * ",
            )
            for iyear, year in enumerate(self.scope["year"]):
                if iyear + 1 == len(self.scope["year"]):
                    end_str = "\n * "
                else:
                    end_str = "\n \t * "

                pct = np.round(
                    self.fuel_blends["hydrogen"]["secondary"]["share"][iyear] * 100,
                    0,
                )
                print(
                    f"in {year} _________________________________________ {pct}%",
                    end=end_str,
                )

                # Primary fuel share

                ind_a = self.find_inputs_indices(
                    must_contain=["transport, passenger bus, ", str(year), "FCEV"],
                )
                ind_array = [
                    x for x in self.get_index_vehicle_from_array(year) if x in index
                ]

                self.a_matrix[
                    :,
                    self.find_inputs_indices(
                        must_contain=["fuel supply for hydrogen vehicles", str(year)],
                    ),
                    ind_a,
                ] = (
                    array[self.array_inputs["fuel mass"], :, ind_array]
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / (array[self.array_inputs["average passengers"], :, ind_array])
                    * -1
                ).T

        if "ICEV-g" in self.scope["powertrain"]:
            index = self.get_index_vehicle_from_array("ICEV-g")

            print(
                f"{self.fuel_blends['cng']['primary']['type']} is completed by {self.fuel_blends['cng']['secondary']['type']}.",
                end="\n \t * ",
            )

            for iyear, year in enumerate(self.scope["year"]):
                if iyear + 1 == len(self.scope["year"]):
                    end_str = "\n * "
                else:
                    end_str = "\n \t * "

                pct = np.round(
                    self.fuel_blends["cng"]["secondary"]["share"][iyear] * 100,
                    0,
                )
                print(
                    f"in {year} _________________________________________ {pct}%",
                    end=end_str,
                )

                # Primary fuel share

                ind_a = self.find_inputs_indices(
                    must_contain=["transport, passenger bus, ", str(year), "ICEV-g"],
                )
                ind_array = [
                    x for x in self.get_index_vehicle_from_array(year) if x in index
                ]

                # Includes pump-to-tank gas leakage, as a fraction of gas input
                self.a_matrix[
                    :,
                    self.find_inputs_indices(
                        must_contain=["fuel supply for gas vehicles", str(year)],
                    ),
                    ind_a,
                ] = (
                    (array[self.array_inputs["fuel mass"], :, ind_array])
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / array[self.array_inputs["average passengers"], :, ind_array]
                    * (
                        1
                        + array[
                            self.array_inputs["CNG pump-to-tank leakage"], :, ind_array
                        ]
                    )
                    * -1
                ).T

                # Gas leakage emission as methane
                self.a_matrix[
                    :,
                    self.inputs[
                        (
                            "Methane, fossil",
                            ("air",),
                            "kilogram",
                        )
                    ],
                    ind_a,
                ] = (
                    (array[self.array_inputs["fuel mass"], :, ind_array])
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / array[self.array_inputs["average passengers"], :, ind_array]
                    * array[self.array_inputs["CNG pump-to-tank leakage"], :, ind_array]
                    * -1
                ).T

                # Fuel-based emissions from CNG, CO2
                # The share and CO2 emissions factor of CNG is retrieved, if used

                share_fossil = 0
                co2_fossil = 0

                if self.fuel_blends["cng"]["primary"]["type"] == "cng":
                    share_fossil += self.fuel_blends["cng"]["primary"]["share"][iyear]
                    co2_fossil = (
                        self.fuel_blends["cng"]["primary"]["CO2"]
                        * self.fuel_blends["cng"]["primary"]["share"][iyear]
                    )

                if self.fuel_blends["cng"]["secondary"]["type"] == "cng":
                    share_fossil += self.fuel_blends["cng"]["secondary"]["share"][iyear]
                    co2_fossil = (
                        self.fuel_blends["cng"]["primary"]["CO2"]
                        * self.fuel_blends["cng"]["secondary"]["share"][iyear]
                    )

                self.a_matrix[
                    :,
                    self.inputs[("Carbon dioxide, fossil", ("air",), "kilogram")],
                    ind_a,
                ] = (
                    (array[self.array_inputs["fuel mass"], :, ind_array] * co2_fossil)
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / array[self.array_inputs["average passengers"], :, ind_array]
                    * -1
                ).T

                # Fuel-based CO2 emission from alternative cng
                # The share of non-fossil gas in the blend is retrieved
                # As well as the CO2 emission factor of the fuel

                share_non_fossil = 0
                co2_non_fossil = 0

                if self.fuel_blends["cng"]["primary"]["type"] != "cng":
                    share_non_fossil += self.fuel_blends["cng"]["primary"]["share"][
                        iyear
                    ]
                    co2_non_fossil = (
                        share_non_fossil * self.fuel_blends["cng"]["primary"]["CO2"]
                    )

                if self.fuel_blends["cng"]["secondary"]["type"] != "cng":
                    share_non_fossil += self.fuel_blends["cng"]["secondary"]["share"][
                        iyear
                    ]
                    co2_non_fossil += (
                        self.fuel_blends["cng"]["secondary"]["share"][iyear]
                        * self.fuel_blends["cng"]["secondary"]["CO2"]
                    )

                self.a_matrix[
                    :,
                    self.inputs[
                        (
                            "Carbon dioxide, from soil or biomass stock",
                            ("air",),
                            "kilogram",
                        )
                    ],
                    ind_a,
                ] = (
                    (
                        (
                            array[self.array_inputs["fuel mass"], :, ind_array]
                            * co2_non_fossil
                        )
                    )
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / array[self.array_inputs["average passengers"], :, ind_array]
                    * -1
                ).T

        if [i for i in self.scope["powertrain"] if i in ["ICEV-d", "HEV-d"]]:
            index = self.get_index_vehicle_from_array(["ICEV-d", "HEV-d"])

            print(
                f"{self.fuel_blends['diesel']['primary']['type']} is completed by {self.fuel_blends['diesel']['secondary']['type']}.",
                end="\n \t * ",
            )

            for iyear, year in enumerate(self.scope["year"]):
                if iyear + 1 == len(self.scope["year"]):
                    end_str = "\n * "
                else:
                    end_str = "\n \t * "

                pct = np.round(
                    self.fuel_blends["diesel"]["secondary"]["share"][iyear] * 100,
                    0,
                )
                print(
                    f"in {year} _________________________________________ {pct}%",
                    end=end_str,
                )

                ind_a = self.find_inputs_indices(
                    must_contain=["transport, passenger bus, ", str(year)],
                    must_also_contain=["ICEV-d", "HEV-d"],
                )

                ind_array = [
                    x for x in self.get_index_vehicle_from_array(year) if x in index
                ]

                # Fuel supply
                self.a_matrix[
                    :,
                    self.find_inputs_indices(
                        must_contain=["fuel supply for diesel vehicles", str(year)],
                    ),
                    ind_a,
                ] = (
                    (array[self.array_inputs["fuel mass"], :, ind_array])
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / array[self.array_inputs["average passengers"], :, ind_array]
                    * -1
                ).T

                # Fuel-based CO2 emission from conventional diesel
                share_fossil = 0
                co2_fossil = 0
                if self.fuel_blends["diesel"]["primary"]["type"] == "diesel":
                    share_fossil = self.fuel_blends["diesel"]["primary"]["share"][iyear]
                    co2_fossil = (
                        self.fuel_blends["diesel"]["primary"]["CO2"]
                        * self.fuel_blends["diesel"]["primary"]["share"][iyear]
                    )

                if self.fuel_blends["diesel"]["secondary"]["type"] == "diesel":
                    share_fossil += self.fuel_blends["diesel"]["secondary"]["share"][
                        iyear
                    ]
                    co2_fossil = (
                        self.fuel_blends["diesel"]["secondary"]["CO2"]
                        * self.fuel_blends["diesel"]["secondary"]["share"][iyear]
                    )

                self.a_matrix[
                    :,
                    self.inputs[("Carbon dioxide, fossil", ("air",), "kilogram")],
                    ind_a,
                ] = (
                    (array[self.array_inputs["fuel mass"], :, ind_array] * co2_fossil)
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / array[self.array_inputs["average passengers"], :, ind_array]
                    * -1
                ).T

                # Fuel-based SO2 emissions
                # Sulfur concentration value for a given country, a given year, as concentration ratio

                sulfur_concentration = self.get_sulfur_content(
                    self.country, "diesel", year
                )

                self.a_matrix[
                    :,
                    self.inputs[("Sulfur dioxide", ("air",), "kilogram")],
                    ind_a,
                ] = (
                    (
                        (
                            array[self.array_inputs["fuel mass"], :, ind_array]
                            * share_fossil  # assumes sulfur only present in conventional diesel
                            * sulfur_concentration
                            * (64 / 32)  # molar mass of SO2/molar mass of O2
                        )
                    )
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / array[self.array_inputs["average passengers"], :, ind_array]
                    * -1
                ).T

                share_non_fossil = 0
                co2_non_fossil = 0

                # Fuel-based CO2 emission from alternative diesel
                # The share of non-fossil fuel in the blend is retrieved
                # As well as the CO2 emission factor of the fuel
                if self.fuel_blends["diesel"]["primary"]["type"] != "diesel":
                    share_non_fossil += self.fuel_blends["diesel"]["primary"]["share"][
                        iyear
                    ]
                    co2_non_fossil = (
                        share_non_fossil * self.fuel_blends["diesel"]["primary"]["CO2"]
                    )

                if self.fuel_blends["diesel"]["secondary"]["type"] != "diesel":
                    share_non_fossil += self.fuel_blends["diesel"]["secondary"][
                        "share"
                    ][iyear]
                    co2_non_fossil += (
                        self.fuel_blends["diesel"]["secondary"]["share"][iyear]
                        * self.fuel_blends["diesel"]["secondary"]["CO2"]
                    )

                self.a_matrix[
                    :,
                    self.inputs[
                        (
                            "Carbon dioxide, from soil or biomass stock",
                            ("air",),
                            "kilogram",
                        )
                    ],
                    ind_a,
                ] = (
                    (
                        (
                            array[self.array_inputs["fuel mass"], :, ind_array]
                            * co2_non_fossil
                        )
                    )
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / array[self.array_inputs["average passengers"], :, ind_array]
                    * -1
                ).T

        # Non-exhaust emissions
        self.a_matrix[
            :,
            self.inputs[
                (
                    "treatment of road wear emissions, lorry",
                    "RER",
                    "kilogram",
                    "road wear emissions, lorry",
                )
            ],
            -self.number_of_cars :,
        ] = array[self.array_inputs["tire wear emissions"], :] / (
            array[self.array_inputs["average passengers"], :]
        )
        self.a_matrix[
            :,
            self.inputs[
                (
                    "treatment of tyre wear emissions, lorry",
                    "RER",
                    "kilogram",
                    "tyre wear emissions, lorry",
                )
            ],
            -self.number_of_cars :,
        ] = array[self.array_inputs["tire wear emissions"], :] / (
            array[self.array_inputs["average passengers"], :]
        )

        # Brake wear emissions
        # BEVs and other hybrid vehicles only emit 20%
        # of what a combustion engine vehicle emit according to
        # https://link.springer.com/article/10.1007/s11367-014-0792-4

        self.a_matrix[
            :,
            self.inputs[
                (
                    "treatment of brake wear emissions, lorry",
                    "RER",
                    "kilogram",
                    "brake wear emissions, lorry",
                )
            ],
            -self.number_of_cars :,
        ] = array[self.array_inputs["brake wear emissions"], :] / (
            array[self.array_inputs["average passengers"], :]
        )

        ind_a = self.find_inputs_indices(
            must_contain=[
                "transport, passenger bus, ",
            ],
            must_also_contain=[
                "BEV-opp",
                "BEV-depot",
                "BEV-motion",
                "FCEV",
                "HEV-d",
            ],
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "treatment of brake wear emissions, lorry",
                    "RER",
                    "kilogram",
                    "brake wear emissions, lorry",
                )
            ],
            ind_a,
        ] *= 0.2

        # Infrastructure: 5.37e-4 per gross tkm
        self.a_matrix[
            :,
            self.inputs[("market for road", "GLO", "meter-year", "road")],
            -self.number_of_cars :,
        ] = (
            (array[self.array_inputs["driving mass"], :] / 1000)
            * 5.37e-4
            / (array[self.array_inputs["average passengers"], :])
        ) * -1

        # Infrastructure maintenance
        self.a_matrix[
            :,
            self.inputs[
                ("market for road maintenance", "RER", "meter-year", "road maintenance")
            ],
            -self.number_of_cars :,
        ] = (
            1.29e-3 / (array[self.array_inputs["average passengers"], :]) * -1
        )

        # Exhaust emissions
        # Non-fuel based emissions

        self.a_matrix[:, self.index_emissions, -self.number_of_cars :] = (
            array[
                [
                    self.array_inputs[self.map_non_fuel_emissions[self.rev_inputs[x]]]
                    for x in self.index_emissions
                ]
            ]
            / (array[self.array_inputs["average passengers"], :])
            * -1
        ).transpose([1, 0, 2])

        # End-of-life disposal and treatment
        self.a_matrix[
            :,
            self.inputs[
                (
                    "treatment of used bus",
                    "CH",
                    "unit",
                    "used bus",
                )
            ],
            -self.number_of_cars :,
        ] = (
            1
            / array[self.array_inputs["lifetime kilometers"]]
            / array[self.array_inputs["average passengers"]]
        ) * (
            array[self.array_inputs["gross mass"]] / 19000
        )

        # Battery EoL
        self.a_matrix[
            :,
            self.inputs[
                (
                    "market for used Li-ion battery",
                    "GLO",
                    "kilogram",
                    "used Li-ion battery",
                )
            ],
            -self.number_of_cars :,
        ] = (
            (
                array[self.array_inputs["energy battery mass"]]
                * (1 + array[self.array_inputs["battery lifetime replacements"]])
            )
            / array[self.array_inputs["lifetime kilometers"]]
            / array[self.array_inputs["average passengers"]]
        )

        # Noise emissions
        self.a_matrix[:, self.index_noise, -self.number_of_cars :] = (
            array[
                [
                    self.array_inputs[self.map_noise_emissions[self.rev_inputs[x]]]
                    for x in self.index_noise
                ]
            ]
            / (array[self.array_inputs["average passengers"], :])
            * -1
        ).transpose([1, 0, 2])

        # Emissions of air conditioner refrigerant r134a
        # Leakage assumed to amount to 16 kg per lifetime according to
        # https://treeze.ch/fileadmin/user_upload/downloads/Publications/Case_Studies/Mobility/544-LCI-Road-NonRoad-Transport-Services-v2.0.pdf

        self.a_matrix[
            :,
            self.inputs[
                ("Ethane, 1,1,1,2-tetrafluoro-, HFC-134a", ("air",), "kilogram")
            ],
            -self.number_of_cars :,
        ] = (
            16
            / self.array.values[self.array_inputs["lifetime kilometers"]]
            / self.array.values[self.array_inputs["average passengers"]]
            * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                ("market for refrigerant R134a", "GLO", "kilogram", "refrigerant R134a")
            ],
            -self.number_of_cars :,
        ] = (
            (16 + 7.5)
            / self.array.values[self.array_inputs["lifetime kilometers"]]
            / self.array.values[self.array_inputs["average passengers"]]
            * -1
        )

        # Charging infrastructure

        # Plugin BEV buses
        # The charging station has a lifetime of 24 years
        # Hence, we calculate the lifetime of the bus
        # We assume two buses per charging station

        index = self.get_index_vehicle_from_array(
            ["BEV-depot", "PHEV-d"],
        )

        self.a_matrix[
            np.ix_(
                np.arange(self.iterations),
                self.find_inputs_indices(
                    must_contain=[
                        "EV charger, level 3, plugin, 200 kW",
                    ],
                ),
                self.find_inputs_indices(
                    must_contain=[
                        "transport, passenger bus, ",
                    ],
                    must_also_contain=["BEV-depot", "PHEV-d"],
                ),
            )
        ] = (
            -1
            / (
                array[self.array_inputs["kilometers per year"], :, index]
                * array[self.array_inputs["average passengers"], :, index]
                * 2
                * 24
            )
        ).T.reshape(
            self.iterations, 1, -1
        )

        # Opportunity charging BEV buses
        # The charging station has a lifetime of 24 years
        # And 10 buses use it
        # Hence, we calculate the lifetime of the bus

        index = self.get_index_vehicle_from_array(
            ["BEV-opp"],
        )

        self.a_matrix[
            np.ix_(
                np.arange(self.iterations),
                self.find_inputs_indices(
                    must_contain=[
                        "EV charger, level 3, with pantograph, 450 kW",
                    ],
                ),
                self.find_inputs_indices(
                    must_contain=[
                        "transport, passenger bus, ",
                    ],
                    must_also_contain=[
                        "BEV-opp",
                    ],
                ),
            )
        ] = (
            -1
            / (
                array[self.array_inputs["kilometers per year"], :, index]
                * array[self.array_inputs["average passengers"], :, index]
                * 10
                * 24
            )
        ).T.reshape(
            self.iterations, 1, -1
        )

        # In-motion charging BEV buses
        # The overhead lines have a lifetime of 40 years
        # And 60 buses use it
        # Hence, we calculate the lifetime of the bus

        index = self.get_index_vehicle_from_array(
            ["BEV-motion"],
        )

        self.a_matrix[
            np.ix_(
                np.arange(self.iterations),
                self.find_inputs_indices(
                    must_contain=[
                        "Overhead lines",
                    ],
                ),
                self.find_inputs_indices(
                    must_contain=[
                        "transport, passenger bus, ",
                    ],
                    must_also_contain=[
                        "BEV-motion",
                    ],
                ),
            )
        ] = (
            -1
            / (
                array[self.array_inputs["lifetime kilometers"], :, index]
                * array[self.array_inputs["average passengers"], :, index]
                * 60
                * 40
            )
        ).T.reshape(
            self.iterations, 1, -1
        )

        print("*********************************************************************")

    def set_inputs_in_a_matrix_for_export(self, array):
        """
        Fill-in the a_matrix matrix. Does not return anything. Modifies in place.
        Shape of the a_matrix matrix (values, products, activities).

        :param array: :attr:`array` from :class:`BusModel` class
        """

        # Glider/Frame
        self.a_matrix[
            :,
            self.inputs[
                (
                    "frame, blanks and saddle, for lorry",
                    "RER",
                    "kilogram",
                    "frame, blanks and saddle, for lorry",
                )
            ],
            # TODO: unit=unit
            self.find_inputs_indices(
                must_contain=["Passenger bus, "], excludes=["market"]
            ),
        ] = (
            array[self.array_inputs["glider base mass"]] * -1
        )

        # Suspension + Brakes
        self.a_matrix[
            :,
            self.inputs[
                ("suspension, for lorry", "RER", "kilogram", "suspension, for lorry")
            ],
            self.find_inputs_indices(
                must_contain=["Passenger bus, "], excludes=["market"]
            ),
        ] = (
            array[
                [
                    self.array_inputs["suspension mass"],
                    self.array_inputs["braking system mass"],
                ],
                :,
            ].sum(axis=0)
            * -1
        )

        # Wheels and tires
        self.a_matrix[
            :,
            self.inputs[
                (
                    "tires and wheels, for lorry",
                    "RER",
                    "kilogram",
                    "tires and wheels, for lorry",
                )
            ],
            self.find_inputs_indices(
                must_contain=["Passenger bus, "], excludes=["market"]
            ),
        ] = (
            array[self.array_inputs["wheels and tires mass"], :] * -1
        )

        # Exhaust
        self.a_matrix[
            :,
            self.inputs[
                (
                    "exhaust system, for lorry",
                    "RER",
                    "kilogram",
                    "exhaust system, for lorry",
                )
            ],
            self.find_inputs_indices(
                must_contain=["Passenger bus, "], excludes=["market"]
            ),
        ] = (
            array[self.array_inputs["exhaust system mass"], :] * -1
        )

        # Electrical system
        self.a_matrix[
            :,
            self.inputs[
                (
                    "power electronics, for lorry",
                    "RER",
                    "kilogram",
                    "power electronics, for lorry",
                )
            ],
            self.find_inputs_indices(
                must_contain=["Passenger bus, "], excludes=["market"]
            ),
        ] = (
            array[self.array_inputs["electrical system mass"], :] * -1
        )

        # Transmission (52% transmission shaft, 36% gearbox + 12% retarder)
        self.a_matrix[
            :,
            self.inputs[
                (
                    "transmission, for lorry",
                    "RER",
                    "kilogram",
                    "transmission, for lorry",
                )
            ],
            self.find_inputs_indices(
                must_contain=["Passenger bus, "], excludes=["market"]
            ),
        ] = (
            array[self.array_inputs["transmission mass"], :] * 0.52 * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                ("gearbox, for lorry", "RER", "kilogram", "gearbox, for lorry")
            ],
            self.find_inputs_indices(
                must_contain=["Passenger bus, "], excludes=["market"]
            ),
        ] = (
            array[self.array_inputs["transmission mass"], :] * 0.36 * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                ("retarder, for lorry", "RER", "kilogram", "retarder, for lorry")
            ],
            self.find_inputs_indices(
                must_contain=["Passenger bus, "], excludes=["market"]
            ),
        ] = (
            array[self.array_inputs["transmission mass"], :] * 0.12 * -1
        )

        # Other components, for non-electric and hybrid trucks
        index = self.get_index_vehicle_from_array(["ICEV-d", "HEV-d", "ICEV-g"])

        ind_a = self.find_inputs_indices(
            must_contain=["Passenger bus, "],
            must_also_contain=["ICEV-d", "HEV-d", "ICEV-g"],
            excludes=["market"],
        )
        self.a_matrix[
            :,
            self.inputs[
                (
                    "other components, for hybrid electric lorry",
                    "RER",
                    "kilogram",
                    "other components, for hybrid electric lorry",
                )
            ],
            ind_a,
        ] = (array[self.array_inputs["other components mass"], :, index] * -1).T

        # Other components, for electric trucks
        index = self.get_index_vehicle_from_array(
            ["BEV-opp", "BEV-depot", "BEV-motion", "FCEV"]
        )
        ind_a = self.find_inputs_indices(
            must_contain=["Passenger bus, "],
            must_also_contain=["BEV-opp", "BEV-depot", "BEV-motion", "FCEV"],
            excludes=["market"],
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "other components, for electric lorry",
                    "RER",
                    "kilogram",
                    "other components, for electric lorry",
                )
            ],
            ind_a,
        ] = (array[self.array_inputs["other components mass"], :, index] * -1).T

        self.a_matrix[
            :,
            self.inputs[
                ("Glider lightweighting", "GLO", "kilogram", "Glider lightweighting")
            ],
            self.find_inputs_indices(
                must_contain=["Passenger bus, "], excludes=["market"]
            ),
        ] = (
            array[self.array_inputs["lightweighting"], :]
            * array[self.array_inputs["glider base mass"], :]
            * -1
        )

        ind_a = self.find_inputs_indices(
            must_contain=["Passenger bus, "], excludes=["market"]
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "maintenance, bus",
                    "CH",
                    "unit",
                    "maintenance, bus",
                )
            ],
            ind_a,
        ] = -1 * (array[self.array_inputs["gross mass"]] / 19000)

        # Powertrain components

        self.a_matrix[
            :,
            self.inputs[
                (
                    "market for converter, for electric passenger car",
                    "GLO",
                    "kilogram",
                    "converter, for electric passenger car",
                )
            ],
            ind_a,
        ] = (
            array[self.array_inputs["converter mass"], :] * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "market for electric motor, electric passenger car",
                    "GLO",
                    "kilogram",
                    "electric motor, electric passenger car",
                )
            ],
            ind_a,
        ] = (
            array[self.array_inputs["electric engine mass"], :] * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "market for inverter, for electric passenger car",
                    "GLO",
                    "kilogram",
                    "inverter, for electric passenger car",
                )
            ],
            ind_a,
        ] = (
            array[self.array_inputs["inverter mass"], :] * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "market for power distribution unit, for electric passenger car",
                    "GLO",
                    "kilogram",
                    "power distribution unit, for electric passenger car",
                )
            ],
            ind_a,
        ] = (
            array[self.array_inputs["power distribution unit mass"], :] * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "internal combustion engine, for lorry",
                    "RER",
                    "kilogram",
                    "internal combustion engine, for lorry",
                )
            ],
            ind_a,
        ] = (
            array[
                [self.array_inputs[l] for l in ["combustion engine mass"]],
                :,
            ].sum(axis=0)
            * -1
        )

        self.a_matrix[
            :, self.inputs[("Ancillary BoP", "GLO", "kilogram", "Ancillary BoP")], ind_a
        ] = array[self.array_inputs["fuel cell ancillary BoP mass"], :] * (
            1 + array[self.array_inputs["fuel cell lifetime replacements"]] * -1
        )

        self.a_matrix[
            :, self.inputs[("Essential BoP", "GLO", "kilogram", "Essential BoP")], ind_a
        ] = (
            array[self.array_inputs["fuel cell essential BoP mass"], :]
            * (1 + array[self.array_inputs["fuel cell lifetime replacements"]])
            * -1
        )

        self.a_matrix[:, self.inputs[("Stack", "GLO", "kilowatt", "Stack")], ind_a] = (
            array[self.array_inputs["fuel cell stack mass"], :]
            * (1 + array[self.array_inputs["fuel cell lifetime replacements"]])
            / 1.02
            * -1
        )

        # Start of printout

        print(
            "****************** IMPORTANT BACKGROUND PARAMETERS ******************",
            end="\n * ",
        )

        # Energy storage for electric trucks

        print(
            f"The country of use is {self.country}",
            end="\n * ",
        )

        # Battery BoP
        index = self.get_index_vehicle_from_array(
            ["BEV-depot", "BEV-opp", "BEV-motion", "FCEV", "HEV-d"]
        )

        ind_a = self.find_inputs_indices(
            must_contain=["Passenger bus, "],
            must_also_contain=["BEV-depot", "BEV-opp", "BEV-motion", "FCEV", "HEV-d"],
            excludes=["market"],
        )

        self.a_matrix[
            :, self.inputs[("Battery BoP", "GLO", "kilogram", "Battery BoP")], ind_a
        ] = (
            (
                array[self.array_inputs["battery BoP mass"], :, index]
                * (
                    1
                    + array[
                        self.array_inputs["battery lifetime replacements"], :, index
                    ]
                )
            )
            * -1
        ).T

        # Zero out electricity requirement for battery cell manufacture
        for battery_tech in ["NMC-111", "NCA", "LFP", "LTO"]:
            battery_cell_label = (
                f"Battery cell, {battery_tech}",
                "GLO",
                "kilogram",
                "Battery cell",
            )

            # Set an input of electricity, given the country of manufacture
            self.a_matrix[
                :,
                self.inputs[
                    (
                        "market group for electricity, medium voltage",
                        "World",
                        "kilowatt hour",
                        "electricity, medium voltage",
                    )
                ],
                self.inputs[battery_cell_label],
            ] = 0

        # We look into `background_configuration` to see what battery chemistry to
        # use for each bus
        for veh in self.background_configuration["energy storage"]["electric"]:
            if veh != "origin" and veh in self.scope["powertrain"]:
                battery_tech = self.background_configuration["energy storage"][
                    "electric"
                ][veh]
                battery_cell_label = (
                    f"Battery cell, {battery_tech}",
                    "GLO",
                    "kilogram",
                    "Battery cell",
                )

                idx = self.get_index_vehicle_from_array([veh])

                ind_a = self.find_inputs_indices(
                    must_contain=["Passenger bus, "],
                    must_also_contain=[veh],
                    excludes=["market"],
                )

                self.a_matrix[:, self.inputs[battery_cell_label], ind_a] = (
                    (
                        array[self.array_inputs["battery cell mass"], :, idx]
                        * (
                            1
                            + array[
                                self.array_inputs["battery lifetime replacements"],
                                :,
                                idx,
                            ]
                        )
                    )
                    * -1
                ).T

                for year in self.scope["year"]:
                    idx = self.get_index_vehicle_from_array(year, veh, method="and")

                    self.a_matrix[
                        np.ix_(
                            np.arange(self.iterations),
                            self.find_inputs_indices(
                                must_contain=[
                                    "electricity market for energy storage production",
                                    str(year),
                                ],
                            ),
                            self.find_inputs_indices(
                                must_contain=["Passenger bus, ", str(year), veh],
                            ),
                        )
                    ] = (
                        array[
                            self.array_inputs["battery cell production electricity"],
                            :,
                            idx,
                        ].T
                        * self.a_matrix[
                            :,
                            self.inputs[battery_cell_label],
                            self.find_inputs_indices(
                                must_contain=["Passenger bus, ", str(year), veh],
                            ),
                        ]
                    ).reshape(
                        self.iterations, 1, -1
                    )

        # Use the inventory of Wolff et al. 2020 for lead acid battery for non-electric and non-hybrid trucks

        ind_a = self.find_inputs_indices(
            must_contain=["Passenger bus, "], must_also_contain=["ICEV-d", "ICEV-g"]
        )

        index = self.get_index_vehicle_from_array(["ICEV-d", "ICEV-g"])

        self.a_matrix[
            :,
            self.inputs[
                (
                    "lead acid battery, for lorry",
                    "RER",
                    "kilogram",
                    "lead acid battery, for lorry",
                )
            ],
            ind_a,
        ] = (
            (
                array[self.array_inputs["battery BoP mass"], :, index]
                * (
                    1
                    + array[
                        self.array_inputs["battery lifetime replacements"], :, index
                    ]
                )
            )
            * -1
        ).T

        self.a_matrix[
            :,
            self.inputs[
                (
                    "lead acid battery, for lorry",
                    "RER",
                    "kilogram",
                    "lead acid battery, for lorry",
                )
            ],
            ind_a,
        ] += (
            (
                array[self.array_inputs["battery cell mass"], :, index]
                * (
                    1
                    + array[
                        self.array_inputs["battery lifetime replacements"], :, index
                    ]
                )
            )
            * -1
        ).T

        # Fuel tank for diesel trucks
        ind_a = self.find_inputs_indices(
            must_contain=["Passenger bus, "], must_also_contain=["ICEV-d", "HEV-d"]
        )
        index = self.get_index_vehicle_from_array(["ICEV-d", "HEV-d"])

        self.a_matrix[
            :,
            self.inputs[
                ("fuel tank, for diesel vehicle", "RER", "kilogram", "fuel tank")
            ],
            ind_a,
        ] = (array[self.array_inputs["fuel tank mass"], :, index] * -1).T

        ind_a = self.find_inputs_indices(
            must_contain=["Passenger bus, "], must_also_contain=["ICEV-g"]
        )

        index = self.get_index_vehicle_from_array("ICEV-g")
        self.a_matrix[
            :,
            self.inputs[
                (
                    "Fuel tank, compressed hydrogen gas, 700bar, with aluminium liner",
                    "RER",
                    "kilogram",
                    "Hydrogen tank",
                )
            ],
            ind_a,
        ] = (array[self.array_inputs["fuel tank mass"], :, index] * -1).T

        if "hydrogen" in self.background_configuration["energy storage"]:
            # If a customization dict is passed
            hydro_tank_technology = self.background_configuration["energy storage"][
                "hydrogen"
            ]["type"]
        else:
            hydro_tank_technology = "carbon fiber"

        dict_tank_map = {
            "carbon fiber": (
                "Fuel tank, compressed hydrogen gas, 700bar",
                "GLO",
                "kilogram",
                "Fuel tank, compressed hydrogen gas, 700bar",
            ),
            "hdpe": (
                "Fuel tank, compressed hydrogen gas, 700bar, with HDPE liner",
                "RER",
                "kilogram",
                "Hydrogen tank",
            ),
            "aluminium": (
                "Fuel tank, compressed hydrogen gas, 700bar, with aluminium liner",
                "RER",
                "kilogram",
                "Hydrogen tank",
            ),
        }

        ind_a = self.find_inputs_indices(
            must_contain=["Passenger bus, "], must_also_contain=["FCEV"]
        )

        index = self.get_index_vehicle_from_array("FCEV")
        self.a_matrix[
            :,
            self.inputs[dict_tank_map[hydro_tank_technology]],
            ind_a,
        ] = (array[self.array_inputs["fuel tank mass"], :, index] * -1).T

        # End-of-life disposal and treatment
        self.a_matrix[
            :,
            self.inputs[
                (
                    "treatment of used bus",
                    "CH",
                    "unit",
                    "used bus",
                )
            ],
            -self.number_of_cars :,
        ] = 1 * (array[self.array_inputs["gross mass"]] / 19000)

        # END of vehicle building

        self.a_matrix[
            :,
            self.find_inputs_indices(
                must_contain=["Passenger bus, "],
            ),
            self.find_inputs_indices(
                must_contain=["transport, passenger bus, "], excludes=["market"]
            ),
        ] = (
            -1
            / array[self.array_inputs["lifetime kilometers"]]
            / (array[self.array_inputs["average passengers"]])
        )

        try:
            sum_renew, co2_intensity_tech = self.define_renewable_rate_in_mix()
        except:
            sum_renew = [0] * len(self.scope["year"])
            co2_intensity_tech = [0] * len(self.scope["year"])

        for iyear, year in enumerate(self.scope["year"]):

            if iyear + 1 == len(self.scope["year"]):
                end_str = "\n * "
            else:
                end_str = "\n \t * "

            renew_rate = np.round(sum_renew[iyear] * 100, 0)
            ghg_rate = int(np.sum(co2_intensity_tech[iyear] * self.mix[iyear]))
            print(
                f"in {year}, % of renewable: {renew_rate}, GHG intensity per kWh: {ghg_rate} g. CO2-eq.",
                end=end_str,
            )

            if any(
                True
                for x in ["BEV-depot", "BEV-opp", "BEV-motion", "PHEV-d"]
                if x in self.scope["powertrain"]
            ):

                index = self.get_index_vehicle_from_array(
                    ["BEV-depot", "BEV-opp", "BEV-motion", "PHEV-d"], year, method="and"
                )

                self.a_matrix[
                    np.ix_(
                        np.arange(self.iterations),
                        self.find_inputs_indices(
                            must_contain=[
                                "electricity supply for electric vehicles",
                                str(year),
                            ]
                        ),
                        self.find_inputs_indices(
                            must_contain=["transport, passenger bus, ", str(year)],
                            must_also_contain=[
                                "BEV-depot",
                                "BEV-opp",
                                "BEV-motion",
                                "PHEV-d",
                            ],
                            excludes=["market"],
                        ),
                    )
                ] = (
                    array[self.array_inputs["electricity consumption"], :, index]
                    / (array[self.array_inputs["average passengers"], :, index])
                    * -1
                ).T.reshape(
                    self.iterations, 1, -1
                )

        # add the diesel consumption from the generator for BEV-motion buses
        # anterior to 2020
        if any(True for x in ["BEV-motion"] if x in self.scope["powertrain"]):
            index = self.get_index_vehicle_from_array(
                ["BEV-motion"], [2000, 2010], method="and"
            )

            self.a_matrix[
                np.ix_(
                    np.arange(self.iterations),
                    self.find_inputs_indices(
                        must_contain=[
                            "diesel, burned in diesel-electric generating set, 18.5kW"
                        ]
                    ),
                    self.find_inputs_indices(
                        must_contain=["transport, passenger bus, ", "BEV-motion"],
                        must_also_contain=[str(2000), str(2010)],
                    ),
                )
            ] = (
                array[self.array_inputs["oxidation energy stored"], :, index]
                / array[self.array_inputs["daily distance"], :, index]
                / array[self.array_inputs["average passengers"], :, index]
                * -1
            ).T.reshape(
                self.iterations, 1, -1
            )

        if "FCEV" in self.scope["powertrain"]:

            index = self.get_index_vehicle_from_array("FCEV")

            print(
                f"{self.fuel_blends['hydrogen']['primary']['type']} is completed by {self.fuel_blends['hydrogen']['secondary']['type']}.",
                end="\n \t * ",
            )
            for iyear, year in enumerate(self.scope["year"]):
                if iyear + 1 == len(self.scope["year"]):
                    end_str = "\n * "
                else:
                    end_str = "\n \t * "

                pct = np.round(
                    self.fuel_blends["hydrogen"]["secondary"]["share"][iyear] * 100,
                    0,
                )
                print(
                    f"in {year} _________________________________________ {pct}%",
                    end=end_str,
                )

                # Primary fuel share

                ind_a = self.find_inputs_indices(
                    must_contain=["transport, passenger bus, ", "FCEV", str(year)]
                )

                ind_array = [
                    x for x in self.get_index_vehicle_from_array(year) if x in index
                ]

                self.a_matrix[
                    :,
                    self.find_inputs_indices(
                        must_contain=["fuel supply for hydrogen vehicles", str(year)]
                    ),
                    ind_a,
                ] = (
                    array[self.array_inputs["fuel mass"], :, ind_array]
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / (array[self.array_inputs["average passengers"], :, ind_array])
                    * -1
                ).T

        if "ICEV-g" in self.scope["powertrain"]:
            index = self.get_index_vehicle_from_array("ICEV-g")

            print(
                f"{self.fuel_blends['cng']['primary']['type']} is completed by {self.fuel_blends['cng']['secondary']['type']}.",
                end="\n \t * ",
            )

            for iyear, year in enumerate(self.scope["year"]):
                if iyear + 1 == len(self.scope["year"]):
                    end_str = "\n * "
                else:
                    end_str = "\n \t * "

                pct = np.round(
                    self.fuel_blends["cng"]["secondary"]["share"][iyear] * 100,
                    0,
                )
                print(
                    f"in {year} _________________________________________ {pct}%",
                    end=end_str,
                )

                # Primary fuel share

                ind_a = self.find_inputs_indices(
                    must_contain=["transport, passenger bus, ", "ICEV-g", str(year)]
                )
                ind_array = [
                    x for x in self.get_index_vehicle_from_array(year) if x in index
                ]

                # Includes pump-to-tank gas leakage, as a fraction of gas input
                self.a_matrix[
                    :,
                    self.find_inputs_indices(
                        must_contain=["fuel supply for gas vehicles", str(year)]
                    ),
                    ind_a,
                ] = (
                    (array[self.array_inputs["fuel mass"], :, ind_array])
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / (array[self.array_inputs["average passengers"], :, ind_array])
                    * (
                        1
                        + array[
                            self.array_inputs["CNG pump-to-tank leakage"], :, ind_array
                        ]
                    )
                    * -1
                ).T

                # Gas leakage emission as methane
                self.a_matrix[
                    :,
                    self.inputs[
                        (
                            "Methane, fossil",
                            ("air",),
                            "kilogram",
                        )
                    ],
                    ind_a,
                ] = (
                    (array[self.array_inputs["fuel mass"], :, ind_array])
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / (array[self.array_inputs["average passengers"], :, ind_array])
                    * array[self.array_inputs["CNG pump-to-tank leakage"], :, ind_array]
                    * -1
                ).T

                # Fuel-based emissions from CNG, CO2
                # The share and CO2 emissions factor of CNG is retrieved, if used

                share_fossil = 0
                co2_fossil = 0

                if self.fuel_blends["cng"]["primary"]["type"] == "cng":
                    share_fossil += self.fuel_blends["cng"]["primary"]["share"][iyear]
                    co2_fossil = (
                        self.fuel_blends["cng"]["primary"]["CO2"]
                        * self.fuel_blends["cng"]["primary"]["share"][iyear]
                    )

                if self.fuel_blends["cng"]["secondary"]["type"] == "cng":
                    share_fossil += self.fuel_blends["cng"]["secondary"]["share"][iyear]
                    co2_fossil = (
                        self.fuel_blends["cng"]["primary"]["CO2"]
                        * self.fuel_blends["cng"]["secondary"]["share"][iyear]
                    )

                self.a_matrix[
                    :,
                    self.inputs[("Carbon dioxide, fossil", ("air",), "kilogram")],
                    ind_a,
                ] = (
                    (array[self.array_inputs["fuel mass"], :, ind_array] * co2_fossil)
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / (array[self.array_inputs["average passengers"], :, ind_array])
                    * -1
                ).T

                # Fuel-based CO2 emission from alternative cng
                # The share of non-fossil gas in the blend is retrieved
                # As well as the CO2 emission factor of the fuel

                share_non_fossil = 0
                co2_non_fossil = 0

                if self.fuel_blends["cng"]["primary"]["type"] != "cng":
                    share_non_fossil += self.fuel_blends["cng"]["primary"]["share"][
                        iyear
                    ]
                    co2_non_fossil = (
                        share_non_fossil
                        * self.fuel_blends["cng"]["primary"]["CO2"]
                        * self.fuel_blends["cng"]["primary"]["share"][iyear]
                    )

                if self.fuel_blends["cng"]["secondary"]["type"] != "cng":
                    share_non_fossil += self.fuel_blends["cng"]["secondary"]["share"][
                        iyear
                    ]
                    co2_non_fossil += (
                        self.fuel_blends["cng"]["secondary"]["share"][iyear]
                        * self.fuel_blends["cng"]["secondary"]["CO2"]
                    )

                self.a_matrix[
                    :,
                    self.inputs[
                        (
                            "Carbon dioxide, from soil or biomass stock",
                            ("air",),
                            "kilogram",
                        )
                    ],
                    ind_a,
                ] = (
                    (
                        (
                            array[self.array_inputs["fuel mass"], :, ind_array]
                            * co2_non_fossil
                        )
                    )
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / (array[self.array_inputs["average passengers"], :, ind_array])
                    * -1
                ).T

        if [i for i in self.scope["powertrain"] if i in ["ICEV-d", "HEV-d"]]:
            index = self.get_index_vehicle_from_array(["ICEV-d", "HEV-d"])

            print(
                f"{self.fuel_blends['diesel']['primary']['type']} is completed by {self.fuel_blends['diesel']['secondary']['type']}.",
                end="\n \t * ",
            )

            for iyear, year in enumerate(self.scope["year"]):
                if iyear + 1 == len(self.scope["year"]):
                    end_str = "\n * "
                else:
                    end_str = "\n \t * "

                pct = np.round(
                    self.fuel_blends["diesel"]["secondary"]["share"][iyear] * 100,
                    0,
                )
                print(
                    f"in {year} _________________________________________ {pct}%",
                    end=end_str,
                )

                ind_a = self.find_inputs_indices(
                    must_contain=["transport, passenger bus, ", str(year)],
                    must_also_contain=["ICEV-d", "HEV-d"],
                )

                ind_array = [
                    x for x in self.get_index_vehicle_from_array(year) if x in index
                ]

                # Fuel supply
                self.a_matrix[
                    :,
                    self.find_inputs_indices(
                        must_contain=["fuel supply for diesel vehicles", str(year)]
                    ),
                    ind_a,
                ] = (
                    (array[self.array_inputs["fuel mass"], :, ind_array])
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / array[self.array_inputs["average passengers"], :, ind_array]
                    * -1
                ).T

                # Fuel-based CO2 emission from conventional diesel
                share_fossil = 0
                co2_fossil = 0
                if self.fuel_blends["diesel"]["primary"]["type"] == "diesel":
                    share_fossil = self.fuel_blends["diesel"]["primary"]["share"][iyear]
                    co2_fossil = (
                        self.fuel_blends["diesel"]["primary"]["CO2"]
                        * self.fuel_blends["diesel"]["primary"]["share"][iyear]
                    )

                if self.fuel_blends["diesel"]["secondary"]["type"] == "diesel":
                    share_fossil += self.fuel_blends["diesel"]["secondary"]["share"][
                        iyear
                    ]
                    co2_fossil += (
                        self.fuel_blends["diesel"]["secondary"]["CO2"]
                        * self.fuel_blends["diesel"]["secondary"]["share"][iyear]
                    )

                self.a_matrix[
                    :,
                    self.inputs[("Carbon dioxide, fossil", ("air",), "kilogram")],
                    ind_a,
                ] = (
                    (array[self.array_inputs["fuel mass"], :, ind_array] * co2_fossil)
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / array[self.array_inputs["average passengers"], :, ind_array]
                    * -1
                ).T

                # Fuel-based SO2 emissions
                # Sulfur concentration value for a given country, a given year, as concentration ratio

                sulfur_concentration = self.get_sulfur_content(
                    self.country, "diesel", year
                )

                self.a_matrix[
                    :,
                    self.inputs[("Sulfur dioxide", ("air",), "kilogram")],
                    ind_a,
                ] = (
                    (
                        (
                            array[self.array_inputs["fuel mass"], :, ind_array]
                            * share_fossil  # assumes sulfur only present in conventional diesel
                            * sulfur_concentration
                            * (64 / 32)  # molar mass of SO2/molar mass of O2
                        )
                    )
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / (array[self.array_inputs["average passengers"], :, ind_array])
                    * -1
                ).T

                share_non_fossil = 0
                co2_non_fossil = 0

                # Fuel-based CO2 emission from alternative diesel
                # The share of non-fossil fuel in the blend is retrieved
                # As well as the CO2 emission factor of the fuel
                if self.fuel_blends["diesel"]["primary"]["type"] != "diesel":
                    share_non_fossil += self.fuel_blends["diesel"]["primary"]["share"][
                        iyear
                    ]
                    co2_non_fossil = (
                        share_non_fossil * self.fuel_blends["diesel"]["primary"]["CO2"]
                    )

                if self.fuel_blends["diesel"]["secondary"]["type"] != "diesel":
                    share_non_fossil += self.fuel_blends["diesel"]["secondary"][
                        "share"
                    ][iyear]
                    co2_non_fossil += (
                        self.fuel_blends["diesel"]["secondary"]["share"][iyear]
                        * self.fuel_blends["diesel"]["secondary"]["CO2"]
                    )

                self.a_matrix[
                    :,
                    self.inputs[
                        (
                            "Carbon dioxide, from soil or biomass stock",
                            ("air",),
                            "kilogram",
                        )
                    ],
                    ind_a,
                ] = (
                    (
                        (
                            array[self.array_inputs["fuel mass"], :, ind_array]
                            * co2_non_fossil
                        )
                    )
                    / array[self.array_inputs["daily distance"], :, ind_array]
                    / (array[self.array_inputs["average passengers"], :, ind_array])
                    * -1
                ).T

        # Non-exhaust emissions
        self.a_matrix[
            :,
            self.inputs[
                (
                    "treatment of road wear emissions, lorry",
                    "RER",
                    "kilogram",
                    "road wear emissions, lorry",
                )
            ],
            self.find_inputs_indices(
                must_contain=["transport, passenger bus, "], excludes=["market"]
            ),
        ] = array[self.array_inputs["tire wear emissions"], :] / (
            array[self.array_inputs["average passengers"], :]
        )
        self.a_matrix[
            :,
            self.inputs[
                (
                    "treatment of tyre wear emissions, lorry",
                    "RER",
                    "kilogram",
                    "tyre wear emissions, lorry",
                )
            ],
            self.find_inputs_indices(
                must_contain=["transport, passenger bus, "], excludes=["market"]
            ),
        ] = array[self.array_inputs["tire wear emissions"], :] / (
            array[self.array_inputs["average passengers"], :]
        )

        # Brake wear emissions
        # BEVs and other hybrid vehicles only emit 20%
        # of what a combustion engine vehicle emit according to
        # https://link.springer.com/article/10.1007/s11367-014-0792-4
        self.a_matrix[
            :,
            self.inputs[
                (
                    "treatment of brake wear emissions, lorry",
                    "RER",
                    "kilogram",
                    "brake wear emissions, lorry",
                )
            ],
            self.find_inputs_indices(
                must_contain=["transport, passenger bus, "], excludes=["market"]
            ),
        ] = array[self.array_inputs["brake wear emissions"], :] / (
            array[self.array_inputs["average passengers"], :]
        )

        ind_a = self.find_inputs_indices(
            must_contain=["transport, passenger bus, "],
            must_also_contain=[
                "BEV-opp",
                "BEV-depot",
                "BEV-motion",
                "FCEV",
                "HEV-d",
                "PHEV-d",
            ],
            excludes=["market"],
        )

        self.a_matrix[
            :,
            self.inputs[
                (
                    "treatment of brake wear emissions, lorry",
                    "RER",
                    "kilogram",
                    "brake wear emissions, lorry",
                )
            ],
            ind_a,
        ] *= 0.2

        # Infrastructure: 5.37e-4 per gross tkm
        self.a_matrix[
            :,
            self.inputs[("market for road", "GLO", "meter-year", "road")],
            self.find_inputs_indices(
                must_contain=["transport, passenger bus, "], excludes=["market"]
            ),
        ] = (
            (array[self.array_inputs["driving mass"], :] / 1000)
            * 5.37e-4
            / (array[self.array_inputs["average passengers"], :])
        ) * -1

        # Infrastructure maintenance
        self.a_matrix[
            :,
            self.inputs[
                ("market for road maintenance", "RER", "meter-year", "road maintenance")
            ],
            self.find_inputs_indices(
                must_contain=["transport, passenger bus, "], excludes=["market"]
            ),
        ] = (
            1.29e-3 / (array[self.array_inputs["average passengers"], :]) * -1
        )

        # Exhaust emissions
        # Non-fuel based emissions

        ind_a = self.find_inputs_indices(
            must_contain=["transport, passenger bus, "],
            must_also_contain=["ICEV-d", "ICEV-g", "HEV-d"],
        )

        ind_array = self.get_index_vehicle_from_array(["ICEV-d", "ICEV-g", "HEV-d"])

        self.a_matrix[
            np.ix_(
                np.arange(self.iterations),
                self.index_emissions,
                ind_a,
            )
        ] = (
            array[
                np.ix_(
                    [
                        self.array_inputs[
                            self.map_non_fuel_emissions[self.rev_inputs[x]]
                        ]
                        for x in self.index_emissions
                    ],
                    np.arange(self.iterations),
                    ind_array,
                )
            ]
            / (array[self.array_inputs["average passengers"], :, ind_array]).T
            * -1
        ).transpose(
            [1, 0, 2]
        )

        # Battery EoL
        self.a_matrix[
            :,
            self.inputs[
                (
                    "market for used Li-ion battery",
                    "GLO",
                    "kilogram",
                    "used Li-ion battery",
                )
            ],
            -self.number_of_cars :,
        ] = array[self.array_inputs["energy battery mass"]] * (
            1 + array[self.array_inputs["battery lifetime replacements"]]
        )

        # Noise emissions
        self.a_matrix[
            np.ix_(
                np.arange(self.iterations),
                self.index_noise,
                self.find_inputs_indices(
                    must_contain=["transport, passenger bus, "], excludes=["market"]
                ),
            )
        ] = (
            array[
                [
                    self.array_inputs[self.map_noise_emissions[self.rev_inputs[x]]]
                    for x in self.index_noise
                ]
            ]
            / (array[self.array_inputs["average passengers"], :])
            * -1
        ).transpose(
            [1, 0, 2]
        )

        # Emissions of air conditioner refrigerant r134a
        # Leakage assumed to amount to 16kg per lifetime according to
        # https://treeze.ch/fileadmin/user_upload/downloads/Publications/Case_Studies/Mobility/544-LCI-Road-NonRoad-Transport-Services-v2.0.pdf

        self.a_matrix[
            :,
            self.inputs[
                ("Ethane, 1,1,1,2-tetrafluoro-, HFC-134a", ("air",), "kilogram")
            ],
            self.find_inputs_indices(
                must_contain=["transport, passenger bus, "], excludes=["market"]
            ),
        ] = (
            16
            / self.array.values[self.array_inputs["lifetime kilometers"]]
            / self.array.values[self.array_inputs["average passengers"]]
            * -1
        )

        self.a_matrix[
            :,
            self.inputs[
                ("market for refrigerant R134a", "GLO", "kilogram", "refrigerant R134a")
            ],
            self.find_inputs_indices(
                must_contain=["transport, passenger bus, "], excludes=["market"]
            ),
        ] = (
            (16 + 7.5)
            / self.array.values[self.array_inputs["lifetime kilometers"]]
            / self.array.values[self.array_inputs["average passengers"]]
            * -1
        )

        # Charging infrastructure

        # Plugin BEV buses
        # The charging station has a lifetime of 24 years
        # Hence, we calculate the lifetime of the bus

        index = self.get_index_vehicle_from_array(
            ["BEV-depot", "PHEV-d"],
        )

        self.a_matrix[
            np.ix_(
                np.arange(self.iterations),
                self.find_inputs_indices(
                    must_contain=["EV charger, level 3, plugin, 200 kW"],
                ),
                self.find_inputs_indices(
                    must_contain=["transport, passenger bus, "],
                    must_also_contain=["BEV-depot", "PHEV-d"],
                ),
            )
        ] = (
            -1
            / (
                array[self.array_inputs["kilometers per year"], :, index]
                * array[self.array_inputs["average passengers"], :, index]
                * 2
                * 24
            )
        ).T.reshape(
            self.iterations, 1, -1
        )

        # Opportunity charging BEV buses
        # The charging station has a lifetime of 24 years
        # And 10 buses use it
        # Hence, we calculate the lifetime of the bus

        index = self.get_index_vehicle_from_array(
            ["BEV-opp"],
        )

        self.a_matrix[
            np.ix_(
                np.arange(self.iterations),
                self.find_inputs_indices(
                    must_contain=["EV charger, level 3, with pantograph, 450 kW"]
                ),
                self.find_inputs_indices(
                    must_contain=["transport, passenger bus, "],
                    must_also_contain=["BEV-opp"],
                ),
            )
        ] = (
            -1
            / (
                array[self.array_inputs["kilometers per year"], :, index]
                * array[self.array_inputs["average passengers"], :, index]
                * 10
                * 24
            )
        ).T.reshape(
            self.iterations, 1, -1
        )

        # In-motion charging BEV buses
        # The overhead lines have a lifetime of 40 years
        # And 30 buses use it
        # Hence, we calculate the lifetime of the bus

        index = self.get_index_vehicle_from_array(
            ["BEV-motion"],
        )

        self.a_matrix[
            np.ix_(
                np.arange(self.iterations),
                self.find_inputs_indices(
                    must_contain=["Overhead lines"],
                ),
                self.find_inputs_indices(
                    must_contain=["transport, passenger bus, "],
                    must_also_contain=["BEV-motion"],
                ),
            )
        ] = (
            -1
            / (
                array[self.array_inputs["lifetime kilometers"], :, index]
                * array[self.array_inputs["average passengers"], :, index]
                * 60
                * 40
            )
        ).T.reshape(
            self.iterations, 1, -1
        )

        print("*********************************************************************")

    def select_heat_supplier(self, heat_supplier):
        """
        The heat supply is an important aspect of direct air capture.
        Here, we can change the supplier of heat.
        :param heat_supplier: by default "waste heat". Must be one of "waste heat", "biomass heat",
        "natural gas heat", "market heat".
        :type heat_supplier: str
        :return:
        """

        d_heat_suppliers = {
            "waste heat": (
                "heat, from municipal waste incineration to generic market for heat district or industrial, other than natural gas",
                "CH",
                "megajoule",
                "heat, district or industrial, other than natural gas",
            ),
            "biomass heat": (
                "heat production, hardwood chips from forest, at furnace 1000kW, state-of-the-art 2014",
                "CH",
                "megajoule",
                "heat, district or industrial, other than natural gas",
            ),
            "natural gas heat": (
                "market group for heat, central or small-scale, natural gas",
                "RER",
                "megajoule",
                "heat, central or small-scale, natural gas",
            ),
            "market heat": (
                "market for heat, from steam, in chemical industry",
                "RER",
                "megajoule",
                "heat, from steam, in chemical industry",
            ),
        }

        air_capture = self.inputs[
            (
                "carbon dioxide, captured from atmosphere",
                "RER",
                "kilogram",
                "carbon dioxide, captured from the atmosphere",
            )
        ]

        methanol_distillation = self.inputs[
            ("Methanol distillation", "RER", "kilogram", "Purified methanol")
        ]

        all_inds = [self.inputs[i] for i in list(d_heat_suppliers.values())]

        # DAC
        heat_amount = self.a_matrix[
            np.ix_(range(self.a_matrix.shape[0]), all_inds, [air_capture])
        ].sum()
        # zero out the heat input
        self.a_matrix[
            np.ix_(range(self.a_matrix.shape[0]), all_inds, [air_capture])
        ] = 0
        # find index of the new supplier and set the amount
        ind = self.inputs[d_heat_suppliers[heat_supplier]]
        self.a_matrix[
            np.ix_(range(self.a_matrix.shape[0]), [ind], [air_capture])
        ] = heat_amount

        # Methanol distillation
        heat_amount = self.a_matrix[
            np.ix_(range(self.a_matrix.shape[0]), all_inds, [methanol_distillation])
        ].sum()
        # zero out the heat input
        self.a_matrix[
            np.ix_(range(self.a_matrix.shape[0]), all_inds, [methanol_distillation])
        ] = 0
        # find index of the new supplier and set the amount
        ind = self.inputs[d_heat_suppliers[heat_supplier]]
        self.a_matrix[
            np.ix_(range(self.a_matrix.shape[0]), [ind], [methanol_distillation])
        ] = heat_amount
