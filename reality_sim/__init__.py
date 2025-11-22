# reality_sim/__init__.py

"""
RealitySim - Библиотека для исследования emerge-тивных свойств реальности
"""

from .core.quantum_fabric import QuantumFabric
from .emergence.laws import EmergentLaws
from .observers.observer import Observer, HUMAN_OBSERVER, LIGO_OBSERVER, ELECTRON_OBSERVER

__version__ = "0.1.0"
__author__ = "Timur Isanov"

__all__ = [
    'QuantumFabric',
    'EmergentLaws', 
    'Observer',
    'HUMAN_OBSERVER',
    'LIGO_OBSERVER', 
    'ELECTRON_OBSERVER'
]
