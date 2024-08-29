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
    assert bm.country == ic.vm.country


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
    )
    results = ic.calculate_impacts()

    assert "13m-urban" not in results.coords["size"].values
    assert "BEV-opp-x" not in results.coords["powertrain"].values


def test_fuel_blend():
    """Test if fuel blends defined by the user are considered"""

    tip = BusInputParameters()
    tip.static()
    _, array = fill_xarray_from_input_parameters(
        tip,
        scope={
            "powertrain": [
                "ICEV-d",
                "ICEV-g",
            ],
            "size": ["13m-city"],
        },
    )
    bm = BusModel(array, country="CH")
    bm.set_all()

    fb = {
        "diesel": {
            "primary": {
                "type": "diesel",
                "share": [0.93, 0.93, 0.93, 0.93, 0.93, 0.93],
            },
            "secondary": {
                "type": "diesel - biodiesel - cooking oil",
                "share": [0.07, 0.07, 0.07, 0.07, 0.07, 0.07],
            },
        },
        "methane": {
            "primary": {
                "type": "methane - biomethane - sewage sludge",
                "share": [1, 1, 1, 1, 1, 1],
            }
        },
    }

    tm = BusModel(array, country="CH", fuel_blend=fb)
    tm.set_all()

    ic = InventoryBus(tm, method="recipe", indicator="midpoint")

    assert np.allclose(
        tm.fuel_blend["diesel"]["primary"]["share"],
        [0.93, 0.93, 0.93, 0.93, 0.93, 0.93],
    )
    assert np.allclose(
        tm.fuel_blend["diesel"]["secondary"]["share"],
        [0.07, 0.07, 0.07, 0.07, 0.07, 0.07],
    )
    assert np.allclose(tm.fuel_blend["methane"]["primary"]["share"], [1, 1, 1, 1, 1, 1])
    assert np.sum(tm.fuel_blend["methane"]["secondary"]["share"]) == 0

    ic.calculate_impacts()

    for fuels in [
        ("diesel", "hydrogen - electrolysis - PEM", "methane"),
        (
            "diesel - biodiesel - palm oil",
            "hydrogen - smr - natural gas",
            "methane - biomethane - sewage sludge",
        ),
        (
            "diesel - biodiesel - rapeseed oil",
            "hydrogen - smr - natural gas with CCS",
            "methane - synthetic - coal",
        ),
        (
            "diesel - biodiesel - cooking oil",
            "hydrogen - wood gasification",
            "methane - synthetic - biological",
        ),
        (
            "diesel - synthetic - FT - coal - economic allocation",
            "hydrogen - atr - biogas",
            "methane - synthetic - biological",
        ),
        (
            "diesel - synthetic - methanol - cement - economic allocation",
            "hydrogen - wood gasification with CCS",
            "methane - synthetic - biological",
        ),
    ]:
        fb = {
            "diesel": {
                "primary": {"type": fuels[0], "share": [1, 1, 1, 1, 1, 1]},
            },
            "hydrogen": {"primary": {"type": fuels[1], "share": [1, 1, 1, 1, 1, 1]}},
            "methane": {"primary": {"type": fuels[2], "share": [1, 1, 1, 1, 1, 1]}},
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
        bm.energy_storage["origin"] = c
        ic = InventoryBus(bm, method="recipe", indicator="midpoint")

        assert ic.vm.country == bm.country
        assert ic.vm.energy_storage["origin"] == c

        ic.calculate_impacts()


def test_endpoint():
    """Test if the correct impact categories are considered"""
    ic = InventoryBus(bm, method="recipe", indicator="endpoint")
    results = ic.calculate_impacts()
    assert "human toxicity: non-carcinogenic" in [
        i.lower() for i in results.impact_category.values
    ]
    assert len(results.impact_category.values) == 26

    """Test if it errors properly if an incorrect method type is give"""
    with pytest.raises(ValueError) as wrapped_error:
        ic = InventoryBus(bm, method="recipe", indicator="endpint")
        ic.calculate_impacts()
    assert wrapped_error.type == ValueError


def test_sulfur_concentration():
    ic = InventoryBus(bm, method="recipe", indicator="endpoint")
    ic.get_sulfur_content("RER", "diesel")


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

    for b in ("3.9",):
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
    for b in ("3.10",):
        for s in ("brightway2", "simapro"):
            for d in ("file", "string"):
                ic.export_lci(
                    ecoinvent_version=b,
                    format=d,
                    directory="directory",
                    software=s,
                )
