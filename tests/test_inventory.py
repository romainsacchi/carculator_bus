import numpy as np
import pytest

from carculator_bus import *

tip = BusInputParameters()
tip.static()
_, array = fill_xarray_from_input_parameters(tip)
bm = BusModel(array, country="CH")
bm.set_all()


def test_check_country():
    # Ensure that country specified in BusModel equals country in InventoryBus
    ic = InventoryBus(bm)
    assert bm.country == ic.country


def test_electricity_mix():
    # Electricity mix must be equal to 1
    ic = InventoryBus(bm)
    assert np.allclose(np.sum(ic.mix, axis=1), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # If we pass a custom electricity mix, check that it is used
    custom_mix = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    bc = {"custom electricity mix": custom_mix}
    ic = InventoryBus(bm, background_configuration=bc)

    assert np.allclose(ic.mix, custom_mix)


def test_scope():
    """Test if scope works as expected"""
    ic = InventoryBus(
        bm,
        method="recipe",
        indicator="midpoint",
        scope={"powertrain": ["ICEV-d"], "size": ["9m"]},
    )
    results = ic.calculate_impacts()

    assert "13m-city" not in results.coords["size"].values
    assert "BEV-opp" not in results.coords["powertrain"].values


def test_fuel_blend():
    """Test if fuel blends defined by the user are considered"""

    fb = {
        "diesel": {
            "primary": {
                "type": "diesel",
                "share": [0.93, 0.93, 0.93, 0.93, 0.93, 0.93],
            },
            "secondary": {
                "type": "biodiesel - cooking oil",
                "share": [0.07, 0.07, 0.07, 0.07, 0.07, 0.07],
            },
        },
        "cng": {
            "primary": {
                "type": "biogas - sewage sludge",
                "share": [1, 1, 1, 1, 1, 1],
            }
        },
    }

    tm = BusModel(array, country="CH", fuel_blend=fb)
    tm.set_all()

    ic = InventoryBus(tm, method="recipe", indicator="midpoint")

    assert np.allclose(
        ic.fuel_blends["diesel"]["primary"]["share"],
        [0.93, 0.93, 0.93, 0.93, 0.93, 0.93],
    )
    assert np.allclose(
        ic.fuel_blends["diesel"]["secondary"]["share"],
        [0.07, 0.07, 0.07, 0.07, 0.07, 0.07],
    )
    assert np.allclose(ic.fuel_blends["cng"]["primary"]["share"], [1, 1, 1, 1, 1, 1])
    assert np.sum(ic.fuel_blends["cng"]["secondary"]["share"]) == 0

    ic.calculate_impacts()

    for fuels in [
        ("diesel", "electrolysis", "cng"),
        (
            "biodiesel - palm oil",
            "smr - natural gas",
            "biogas - sewage sludge",
        ),
        (
            "biodiesel - rapeseed oil",
            "smr - natural gas with CCS",
            "biogas - biowaste",
        ),
        (
            "biodiesel - cooking oil",
            "wood gasification with EF with CCS",
            "biogas - biowaste",
        ),
        (
            "biodiesel - algae",
            "atr - biogas",
            "biogas - biowaste",
        ),
        (
            "synthetic diesel - energy allocation",
            "wood gasification with EF with CCS (Swiss forest)",
            "syngas",
        ),
    ]:
        fb = {
            "diesel": {
                "primary": {"type": fuels[0], "share": [1, 1, 1, 1, 1, 1]},
            },
            "hydrogen": {"primary": {"type": fuels[1], "share": [1, 1, 1, 1, 1, 1]}},
            "cng": {"primary": {"type": fuels[2], "share": [1, 1, 1, 1, 1, 1]}},
        }

        tm = BusModel(array, country="CH", fuel_blend=fb)
        tm.set_all()
        ic = InventoryBus(tm, method="recipe", indicator="midpoint")
        ic.calculate_impacts()


def test_countries():
    """Test that calculation works with all countries"""
    for c in [
        "AO",
        "AT",
        "AU",
        "BE",
    ]:
        bm.country = c
        ic = InventoryBus(bm, method="recipe", indicator="midpoint")
        assert (
            ic.background_configuration["energy storage"]["electric"]["origin"]
            == bm.country
        )
        ic.calculate_impacts()


def test_endpoint():
    """Test if the correct impact categories are considered"""
    ic = InventoryBus(bm, method="recipe", indicator="endpoint")
    results = ic.calculate_impacts()
    assert "human health" in [i.lower() for i in results.impact_category.values]
    assert len(results.impact_category.values) == 4

    """Test if it errors properly if an incorrect method type is give"""
    with pytest.raises(TypeError) as wrapped_error:
        ic = InventoryBus(bm, method="recipe", indicator="endpint")
        ic.calculate_impacts()
    assert wrapped_error.type == TypeError


def test_sulfur_concentration():
    ic = InventoryBus(bm, method="recipe", indicator="endpoint")
    ic.get_sulfur_content("RER", "diesel", 2000)
    ic.get_sulfur_content("foo", "diesel", 2000)

    with pytest.raises(ValueError) as wrapped_error:
        ic.get_sulfur_content("FR", "diesel", "jku")
    assert wrapped_error.type == ValueError


def test_custom_electricity_mix():
    """Test if a wrong number of electricity mixes throws an error"""

    bc = {
        "custom electricity mix": [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    }

    with pytest.raises(ValueError) as wrapped_error:
        InventoryBus(
            bm, method="recipe", indicator="endpoint", background_configuration=bc
        )
    assert wrapped_error.type == ValueError

    """ Test if a sum of share superior to 1 throws an error """

    bc = {
        "custom electricity mix": [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    }

    with pytest.raises(ValueError) as wrapped_error:
        InventoryBus(
            bm, method="recipe", indicator="endpoint", background_configuration=bc
        )
    assert wrapped_error.type == ValueError


def test_export_to_bw():
    tip = BusInputParameters()
    tip.static()
    _, array = fill_xarray_from_input_parameters(tip)
    tm = BusModel(array, country="CH")
    tm.set_all()

    """Test that inventories export successfully"""
    ic = InventoryBus(tm, method="recipe", indicator="midpoint")

    for b in ("3.5", "3.6", "3.7", "3.8"):
        ic.export_lci(
            ecoinvent_version=b,
            format="bw2io",
        )


def test_export_to_excel():
    tip = BusInputParameters()
    tip.static()
    _, array = fill_xarray_from_input_parameters(tip)
    tm = BusModel(array, country="CH")
    tm.set_all()

    """Test that inventories export successfully to Excel/CSV"""
    ic = InventoryBus(tm)
    for b in ("3.5", "3.6", "3.7", "3.8"):
        for s in ("brightway2", "simapro"):
            for d in ("file", "string"):
                ic.export_lci(
                    ecoinvent_version=b,
                    format=d,
                    directory="directory",
                    software=s,
                )
