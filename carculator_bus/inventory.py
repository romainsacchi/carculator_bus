"""
inventory.py contains the InventoryBus subclass which inherits from the Inventory class
that provides all methods to solve inventories.
"""

import warnings

import numpy as np
from carculator_utils.inventory import Inventory

from . import DATA_DIR

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

IAM_FILES_DIR = DATA_DIR / "IAM"


class InventoryBus(Inventory):
    """
    Build and solve the inventory for results characterization and inventory export
    """

    def fill_in_A_matrix(self):
        """
        Fill-in the A matrix. Does not return anything. Modifies in place.
        Shape of the A matrix (values, products, activities).

        :attr:`array` from :class:`CarModel` class
        """

        # Assembly
        self.A[
            :,
            self.find_input_indices(("assembly operation, for lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="curb mass") * -1
        )

        # Glider/Frame
        self.A[
            :,
            self.find_input_indices(("frame, blanks and saddle, for lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="glider base mass") * -1
        )

        # Suspension + Brakes
        self.A[
            :,
            self.find_input_indices(("suspension, for lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(
                parameter=[
                    "suspension mass",
                    "braking system mass",
                ]
            ).sum(dim="parameter")
            * -1
        )

        # Wheels and tires
        self.A[
            :,
            self.find_input_indices(("tires and wheels, for lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="wheels and tires mass") * -1
        )

        # Exhaust
        self.A[
            :,
            self.find_input_indices(("exhaust system, for lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="exhaust system mass") * -1
        )

        # Electrical system
        self.A[
            :,
            self.find_input_indices(("power electronics, for lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="electrical system mass") * -1
        )

        # Transmission (52% transmission shaft, 36% gearbox + 12% retarder)
        self.A[
            :,
            self.find_input_indices(("transmission, for lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="transmission mass") * 0.52 * -1
        )

        self.A[
            :,
            self.find_input_indices(("gearbox, for lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="transmission mass") * 0.36 * -1
        )

        self.A[
            :,
            self.find_input_indices(("retarder, for lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="transmission mass") * 0.12 * -1
        )

        # Other components, for non-electric and hybrid trucks
        self.A[
            :,
            self.find_input_indices(("other components, for hybrid electric lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="other components mass")
            * (self.array.sel(parameter="combustion power") > 0)
            * -1
        )

        # Other components, for electric trucks
        self.A[
            :,
            self.find_input_indices(("other components, for electric lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="other components mass")
            * (self.array.sel(parameter="combustion power") == 0)
            * -1
        )

        self.A[
            :,
            self.find_input_indices(("glider lightweighting",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="lightweighting")
            * self.array.sel(parameter="glider base mass")
            * -1
        )

        self.A[
            :,
            self.find_input_indices(("maintenance, bus",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = -1 * (self.array.sel(parameter="gross mass") / 19000)

        # Electric powertrain components
        self.A[
            :,
            self.find_input_indices(
                ("market for converter, for electric passenger car",)
            ),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="converter mass") * -1
        )

        self.A[
            :,
            self.find_input_indices(
                ("market for electric motor, electric passenger car",)
            ),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="electric engine mass") * -1
        )

        self.A[
            :,
            self.find_input_indices(
                ("market for inverter, for electric passenger car",)
            ),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="inverter mass") * -1
        )

        self.A[
            :,
            self.find_input_indices(
                ("market for power distribution unit, for electric passenger car",)
            ),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="power distribution unit mass") * -1
        )

        self.A[
            :,
            self.find_input_indices(("internal combustion engine, for lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            self.array.sel(parameter="combustion engine mass") * -1
        )

        # Energy storage
        self.add_fuel_cell_stack()
        self.add_hydrogen_tank()
        self.add_battery()

        # Use the inventory of Wolff et al. 2020 for
        # lead acid battery for non-electric
        # and non-hybrid trucks
        # replaced every 5 years
        # 16 kg a piece

        self.A[
            :,
            self.find_input_indices(("lead acid battery, for lorry",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = (
            16.0  # kg/battery
            * (
                self.array.sel(parameter="lifetime kilometers")
                / self.array.sel(parameter="kilometers per year")
                / 5  # years
            )
            * (self.array.sel(parameter="combustion power") > 0)
        ) * -1

        # Fuel tank for diesel buses
        self.A[
            :,
            self.find_input_indices(("fuel tank, for diesel vehicle",)),
            [
                j
                for i, j in self.inputs.items()
                if i[0].startswith("bus, ") and "EV-d" in i[0] and "BEV" not in i[0]
            ],
        ] = (
            self.array.sel(
                parameter="fuel tank mass",
                combined_dim=[
                    d
                    for d in self.array.coords["combined_dim"].values
                    if any(x in d for x in ["ICEV-d", "HEV-d"])
                ],
            )
            * -1
        )

        self.add_cng_tank()

        # End-of-life disposal and treatment
        self.A[
            :,
            self.find_input_indices(("treatment of used bus",)),
            [j for i, j in self.inputs.items() if i[0].startswith("bus, ")],
        ] = 1 * (self.array.sel(parameter="gross mass") / 19000)

        # END of vehicle building

        # Add vehicle dataset to transport dataset
        self.add_vehicle_to_transport_dataset()

        self.display_renewable_rate_in_mix()

        self.add_electricity_to_electric_vehicles()

        self.add_hydrogen_to_fuel_cell_vehicles()

        # add the diesel consumption from the generator for BEV-motion buses
        # anterior to 2020
        if any(
            True
            for x in [
                "BEV-motion",
            ]
            if x in self.scope["powertrain"]
        ):
            self.A[
                np.ix_(
                    np.arange(self.iterations),
                    self.find_input_indices(
                        contains=(
                            "diesel, burned in diesel-electric generating set, 18.5kW",
                        ),
                        excludes=("market for",),
                    ),
                    self.find_input_indices(
                        contains=("transport, bus, ", "BEV-motion")
                    ),
                )
            ] = (
                self.array.sel(
                    parameter=[
                        "oxidation energy stored",
                    ],
                    combined_dim=[
                        d
                        for d in self.array.coords["combined_dim"].values
                        if "BEV-motion" in d
                    ],
                ).values
                / self.array.sel(
                    parameter=[
                        "daily distance",
                    ],
                    combined_dim=[
                        d
                        for d in self.array.coords["combined_dim"].values
                        if "BEV-motion" in d
                    ],
                ).values
            ) * -1

        self.add_fuel_to_vehicles("methane", ["ICEV-g"], "EV-g")

        self.A[
            :,
            self.find_input_indices(("fuel supply for methane vehicles",)),
            self.find_input_indices((f"transport, {self.vm.vehicle_type}",)),
        ] *= 1 + self.array.sel(parameter="CNG pump-to-tank leakage")

        # Gas leakage to air
        self.A[
            :,
            self.inputs[("Methane, fossil", ("air",), "kilogram")],
            self.find_input_indices((f"transport, {self.vm.vehicle_type}",)),
        ] *= 1 + self.array.sel(parameter="CNG pump-to-tank leakage")

        self.add_fuel_to_vehicles("diesel", ["ICEV-d", "HEV-d"], "EV-d")

        self.add_abrasion_emissions()

        self.add_road_construction()

        self.add_road_maintenance()

        self.add_exhaust_emissions()

        self.add_noise_emissions()

        self.add_refrigerant_emissions()

        # Charging infrastructure

        # Plugin BEV buses
        # The charging station has a lifetime of 24 years
        # Hence, we calculate the lifetime of the bus

        self.A[
            np.ix_(
                np.arange(self.iterations),
                self.find_input_indices(("EV charger, level 3, plugin, 200 kW",)),
                self.find_input_indices(
                    contains=(f"transport, {self.vm.vehicle_type}", "BEV"),
                    excludes=("motion", "opp"),
                ),
            )
        ] = -1 / (
            self.array.sel(
                parameter=["kilometers per year"],
                combined_dim=[
                    d
                    for d in self.array.coords["combined_dim"].values
                    if "BEV-depot" in d
                ],
            )
            * 2
            * 24
        )

        # Opportunity charging BEV buses
        # The charging station has a lifetime of 24 years
        # And 10 buses use it
        # Hence, we calculate the lifetime of the bus

        self.A[
            np.ix_(
                np.arange(self.iterations),
                self.find_input_indices(
                    ("EV charger, level 3, with pantograph, 450 kW",)
                ),
                self.find_input_indices(
                    contains=(f"transport, {self.vm.vehicle_type}", "BEV-opp")
                ),
            )
        ] = -1 / (
            self.array.sel(
                parameter=["kilometers per year"],
                combined_dim=[
                    d
                    for d in self.array.coords["combined_dim"].values
                    if "BEV-opp" in d
                ],
            )
            * 10
            * 24
        )

        # In-motion charging BEV buses
        # The overhead lines have a lifetime of 40 years
        # And 30 buses use it
        # Hence, we calculate the lifetime of the bus

        if "BEV-motion" in self.scope["powertrain"]:
            self.A[
                np.ix_(
                    np.arange(self.iterations),
                    self.find_input_indices(("catenary system",)),
                    self.find_input_indices(
                        contains=(f"transport, {self.vm.vehicle_type}", "BEV-motion")
                    ),
                )
            ] = (
                -1
                / (
                    self.array.sel(
                        parameter="lifetime kilometers",
                        combined_dim=[
                            d
                            for d in self.array.coords["combined_dim"].values
                            if "BEV-motion" in d
                        ],
                    )
                    * 30
                    * 40
                ).values[:, None]
            )

        print("*********************************************************************")
