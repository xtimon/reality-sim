# reality_sim/observers/observer.py

import numpy as np
from typing import Union

class Observer:
    """
    Модель наблюдателя с определенной 'силой декогеренции'.
    
    Моделирует влияние наблюдателя на квантовую систему через декогеренцию.
    Сила декогеренции зависит от массы, температуры и сложности наблюдателя.
    """
    
    def __init__(self, mass: float, temperature: float, complexity: float):
        """
        Инициализирует наблюдателя.
        
        Args:
            mass: Масса наблюдателя в кг
            temperature: Температура наблюдателя в Кельвинах
            complexity: Мера сложности наблюдателя (безразмерная)
        
        Raises:
            ValueError: Если параметры некорректны
        """
        if mass < 0:
            raise ValueError(f"mass должна быть >= 0, получено {mass}")
        if temperature < 0:
            raise ValueError(f"temperature должна быть >= 0, получено {temperature}")
        if complexity < 0:
            raise ValueError(f"complexity должна быть >= 0, получено {complexity}")
        
        self.mass = mass
        self.temperature = temperature
        self.complexity = complexity
        self.decoherence_power = self._calculate_decoherence_power()
    
    def _calculate_decoherence_power(self) -> float:
        """
        Вычисляет Γ_набл (силу декогеренции) по формуле.
        
        Формула: Γ = ln(1 + |m * c / (T + ε)|)
        где m - масса, c - сложность, T - температура
        
        Returns:
            Сила декогеренции
        """
        base_power = self.mass * self.complexity
        if self.temperature > 0:
            base_power /= (self.temperature + 1e-10)  # Избегаем деления на 0
        return np.log(1 + abs(base_power))
    
    def observe_system(self, quantum_system: Union[float, np.ndarray], 
                      observation_time: float = 1.0) -> Union[float, np.ndarray]:
        """
        Наблюдатель взаимодействует с квантовой системой, вызывая декогеренцию.
        
        Args:
            quantum_system: Квантовая система (число или массив)
            observation_time: Время наблюдения
        
        Returns:
            Система после декогеренции
        
        Raises:
            ValueError: Если observation_time < 0
        """
        if observation_time < 0:
            raise ValueError(f"observation_time должна быть >= 0, получено {observation_time}")
        
        decoherence_rate = self.decoherence_power * observation_time
        coherence = np.exp(-decoherence_rate)
        return self._apply_decoherence(quantum_system, coherence)
    
    def _apply_decoherence(self, system: Union[float, np.ndarray], 
                          coherence_level: float) -> Union[float, np.ndarray]:
        """
        Применяет декогеренцию к системе.
        
        Args:
            system: Квантовая система
            coherence_level: Уровень сохранившейся когерентности (0-1)
        
        Returns:
            Система после декогеренции
        """
        if isinstance(system, np.ndarray):
            return system * coherence_level
        else:
            return system * coherence_level
    
    def __str__(self) -> str:
        """Строковое представление наблюдателя"""
        return (f"Observer(mass={self.mass:.2e} kg, "
                f"T={self.temperature:.2f} K, "
                f"Γ={self.decoherence_power:.2e})")
    
    def __repr__(self) -> str:
        """Представление для отладки"""
        return (f"Observer(mass={self.mass}, "
                f"temperature={self.temperature}, "
                f"complexity={self.complexity})")

# Стандартные наблюдатели
HUMAN_OBSERVER = Observer(mass=70, temperature=300, complexity=1e15)
LIGO_OBSERVER = Observer(mass=40, temperature=0.01, complexity=1e10)
ELECTRON_OBSERVER = Observer(mass=9e-31, temperature=2.7, complexity=1)
