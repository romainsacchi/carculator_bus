import numpy as np

from carculator_bus import *

tip = BusInputParameters()
tip.static()
_, array = fill_xarray_from_input_parameters(tip)
tm = BusModel(array, country="CH")
tm.set_all()


def test_presence_PHEVe():
    # PHEV-e should be dropped
    assert "PHEV-e" not in tm.array.powertrain.values.tolist()


def test_ttw_energy_against_VECTO():
    # The TtW energy consumption of a 13m city bus diesel must be
    # within an interval given by VECTO
    vecto_empty, vecto_full = (8300, 13700)

    assert (
        vecto_empty
        <= tm.array.sel(
            powertrain="ICEV-d",
            year=2020,
            size="13m-city",
            parameter="TtW energy",
            value=0,
        )
        <= vecto_full
    )


# The fuel cell stack mass must be in a given interval


def test_auxiliary_power_demand():
    # The auxilliary power demand must be lower for combustion trucks
    assert np.all(
        tm.array.sel(
            powertrain="ICEV-d", year=2020, parameter="auxiliary power demand", value=0
        )
        < tm.array.sel(
            powertrain="BEV-opp", year=2020, parameter="auxiliary power demand", value=0
        )
    )


def test_battery_replacement():
    # Battery replacements cannot be lower than 0
    assert np.all(tm["battery lifetime replacements"] >= 0)


def test_cargo_mass():
    # Cargo mass cannot be superior to available payload
    assert np.all(tm["curb mass"] <= tm["driving mass"])

    # Cargo mass must equal the available payload * load factor
    # assert np.allclose((tm["available payload"] * tm["capacity utilization"]), tm["total cargo mass"])


def test_electric_utility_factor():
    # Electric utility factor must be between 0 and 1
    assert 0 <= np.all(tm["electric utility factor"]) <= 1


def test_fuel_blends():
    # Shares of a fuel blend must equal 1
    for fuel in tm.fuel_blend:
        np.testing.assert_array_equal(
            np.array(tm.fuel_blend[fuel]["primary"]["share"])
            + np.array(tm.fuel_blend[fuel]["secondary"]["share"]),
            [1, 1, 1, 1, 1, 1],
        )

    # a_matrix fuel cannot be specified both as primary and secondary fuel
    for fuel in tm.fuel_blend:
        assert (
            tm.fuel_blend[fuel]["primary"]["type"]
            != tm.fuel_blend[fuel]["secondary"]["type"]
        )


def test_battery_mass():
    # Battery mass must equal cell mass and BoP mass
    with tm("BEV-opp") as cpm:
        assert np.allclose(
            cpm["energy battery mass"],
            cpm["battery cell mass"] + cpm["battery BoP mass"],
        )

    # Cell mass must equal capacity divided by energy density of cells
    with tm("BEV-depot") as cpm:
        assert np.allclose(
            cpm["battery cell mass"],
            cpm["electric energy stored"] / cpm["battery cell energy density, NMC-111"],
        )


def test_noise_emissions():
    # Noise emissions of a city bus must only affect urban area
    tm = BusModel(array, country="CH")
    tm.set_all()

    list_comp = ["rural", "suburban"]
    params = [
        p
        for p in tm.array.parameter.values
        if "noise" in p and any([x in p for x in list_comp])
    ]

    assert tm.array.sel(size="13m-city", parameter=params).sum() == 0
