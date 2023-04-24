from carculator_utils import get_standard_driving_cycle_and_gradient
import numpy as np


def get_driving_cycle(size: list) -> np.ndarray:
    return get_standard_driving_cycle_and_gradient(
        vehicle_type="bus",
        vehicle_sizes=size,
        name="bus",
    )[0]


def get_road_gradient(size: list) -> np.ndarray:
    return get_standard_driving_cycle_and_gradient(
        vehicle_type="bus",
        vehicle_sizes=size,
        name="bus",
    )[1]