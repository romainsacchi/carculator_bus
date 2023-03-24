from pathlib import Path

import numpy as np
import pandas as pd
from carculator_utils.array import fill_xarray_from_input_parameters

from carculator_bus import BusInputParameters, BusModel

bip = BusInputParameters()
bip.static()
_, array = fill_xarray_from_input_parameters(bip)
bm = BusModel(array)
bm.set_all()

# def test_energy_target_compliance():
# ICEV-d and ICEV-g after 2020 should comply with given energy targets
# In this case, 30% reduction in 2030 compared to 2020
#    assert np.all((bm.array.sel(powertrain=["ICEV-d", "ICEV-g"], size="40t", year=2030, parameter="TtW energy")/
#     bm.array.sel(powertrain=["ICEV-d", "ICEV-g"], size="40t", year=2020, parameter="TtW energy")) <= .7)


def test_fuel_blends():
    # Shares of a fuel blend must equal 1
    for fuel in bm.fuel_blend:
        np.testing.assert_array_equal(
            np.array(bm.fuel_blend[fuel]["primary"]["share"])
            + np.array(bm.fuel_blend[fuel]["secondary"]["share"]),
            [1, 1, 1, 1, 1, 1],
        )

    # A fuel cannot be specified both as primary and secondary fuel
    for fuel in bm.fuel_blend:
        assert (
            bm.fuel_blend[fuel]["primary"]["type"]
            != bm.fuel_blend[fuel]["secondary"]["type"]
        )


def test_battery_mass():
    # Battery mass must equal cell mass and BoP mass
    assert np.allclose(
        bm.array.sel(
            parameter="energy battery mass",
            powertrain="BEV-depot",
            year=2020,
            size="13m-city",
        ),
        bm.array.sel(
            parameter="battery cell mass",
            powertrain="BEV-depot",
            year=2020,
            size="13m-city",
        )
        + bm.array.sel(
            parameter="battery BoP mass",
            powertrain="BEV-depot",
            year=2020,
            size="13m-city",
        ),
    )

    # Cell mass must equal capacity divided by energy density of cells
    assert np.allclose(
        bm.array.sel(
            parameter="battery cell mass",
            powertrain="BEV-depot",
            year=2020,
            size="13m-city",
        ),
        bm.array.sel(
            parameter="electric energy stored",
            powertrain="BEV-depot",
            year=2020,
            size="13m-city",
        )
        / bm.array.sel(
            parameter="battery cell energy density",
            powertrain="BEV-depot",
            year=2020,
            size="13m-city",
        ),
    )


DATA = Path(__file__, "..").resolve() / "fixtures" / "bus_values.xlsx"
OUTPUT = Path(__file__, "..").resolve() / "fixtures" / "test_model_results.xlsx"
ref = pd.read_excel(DATA, index_col=0)


def test_model_results():
    list_powertrains = [
        "ICEV-d",
        "BEV-depot",
        "BEV-opp",
        "BEV-motion",
        "ICEV-g",
        "HEV-d",
        "FCEV",
    ]
    list_sizes = ["13m-city", "18m"]
    list_years = [
        2020,
    ]

    l_res = []

    for pwt in list_powertrains:
        for size in list_sizes:
            for year in list_years:
                for param in bm.array.parameter.values:
                    val = float(
                        bm.array.sel(
                            powertrain=pwt,
                            size=size,
                            year=year,
                            parameter=param,
                            value=0,
                        ).values
                    )

                    try:
                        ref_val = (
                            ref.loc[
                                (ref["powertrain"] == pwt)
                                & (ref["size"] == size)
                                & (ref["parameter"] == param),
                                year,
                            ]
                            .values.astype(float)
                            .item(0)
                        )
                    except:
                        ref_val = 1

                    _ = lambda x: np.where(ref_val == 0, 1, ref_val)
                    diff = val / _(ref_val)
                    l_res.append([pwt, size, year, param, val, ref_val, diff])

    pd.DataFrame(
        l_res,
        columns=["powertrain", "size", "year", "parameter", "val", "ref_val", "diff"],
    ).to_excel(OUTPUT)
