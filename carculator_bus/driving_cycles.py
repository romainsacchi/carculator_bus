import numpy as np
from carculator_utils import get_standard_driving_cycle_and_gradient


def get_driving_cycle(size: list) -> np.ndarray:
    """
    Get driving cycle.

    :param size: List of vehicle sizes.
    :returns: :meth:`ndarray` object
    """
    return get_standard_driving_cycle_and_gradient(
        vehicle_type="bus",
        vehicle_sizes=size,
        name="bus",
    )[0]


def get_road_gradient(size: list) -> np.ndarray:
    """
    Get road gradient data.

    :param size: List of vehicle sizes.
    :returns: :meth:`ndarray` object
    """
    return get_standard_driving_cycle_and_gradient(
        vehicle_type="bus",
        vehicle_sizes=size,
        name="bus",
    )[1]
