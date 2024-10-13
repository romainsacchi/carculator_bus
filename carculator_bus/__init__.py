"""

Submodules
==========

.. autosummary::
    :toctree: _autosummary


"""

__all__ = (
    "BusInputParameters",
    "fill_xarray_from_input_parameters",
    "BusModel",
    "InventoryBus",
    "get_driving_cycle",
    "get_road_gradient",
)

# library version
__version__ = (0, 1, 0)

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

from carculator_utils.array import fill_xarray_from_input_parameters

from .bus_input_parameters import BusInputParameters
from .driving_cycles import get_driving_cycle, get_road_gradient
from .inventory import InventoryBus
from .model import BusModel
