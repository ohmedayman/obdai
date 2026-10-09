"""
Synthetic Data Generation Module for OBD-II Electrical Telemetry
"""
from .synthetic_generator import (
    OBDSyntheticDataGenerator,
    SimulationPattern,
    VehicleElectricalConfig,
)

__all__ = [
    "OBDSyntheticDataGenerator",
    "SimulationPattern",
    "VehicleElectricalConfig",
]
