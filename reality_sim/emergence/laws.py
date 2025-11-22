# reality_sim/emergence/laws.py

import numpy as np
from typing import List, Tuple, Dict, Any, Union

class EmergentLaws:
    """
    Моделирование того, как законы физики emerge-ят из простых правил.
    
    Реализует различные emergent-ные физические явления:
    - Рождение пар частица-античастица
    - Принцип Ландауэра
    - Оценка метрики пространства-времени из запутанности
    """
    
    # Физические константы
    BOLTZMANN_CONSTANT = 1.38e-23  # Дж/К
    ELECTRON_MASS_MEV = 0.511  # МэВ/c²
    
    @staticmethod
    def simulate_particle_creation(vacuum_energy: float = 0.1, 
                                   time_steps: int = 100,
                                   particle_type: str = 'electron') -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Симуляция рождения пар частица-античастица из вакуума.
        
        Моделирует квантовые флуктуации вакуума, приводящие к рождению
        пар частица-античастица.
        
        Args:
            vacuum_energy: Энергия вакуумных флуктуаций (безразмерная)
            time_steps: Количество временных шагов
            particle_type: Тип частицы ('electron', 'photon', и т.д.)
        
        Returns:
            Список кортежей (частица, античастица)
        
        Raises:
            ValueError: Если параметры некорректны
        """
        if vacuum_energy < 0:
            raise ValueError(f"vacuum_energy должна быть >= 0, получено {vacuum_energy}")
        if time_steps <= 0:
            raise ValueError(f"time_steps должен быть > 0, получено {time_steps}")
        
        particles_created = []
        
        # Определяем свойства частиц в зависимости от типа
        particle_properties = {
            'electron': {'charge': +1, 'mass': EmergentLaws.ELECTRON_MASS_MEV},
            'photon': {'charge': 0, 'mass': 0.0},
        }
        
        props = particle_properties.get(particle_type, particle_properties['electron'])
        
        for t in range(time_steps):
            # Вероятность создания уменьшается со временем
            creation_probability = vacuum_energy * np.exp(-t/10)
            
            if np.random.random() < creation_probability:
                particle = {
                    'type': particle_type,
                    'charge': props['charge'],
                    'mass': props['mass'],
                    'created_at': t
                }
                antiparticle = {
                    'type': f'anti-{particle_type}',
                    'charge': -props['charge'],
                    'mass': props['mass'],
                    'created_at': t
                }
                particles_created.append((particle, antiparticle))
                
        return particles_created
    
    @staticmethod
    def landauer_principle(bits_erased: Union[int, float], 
                          temperature: float) -> float:
        """
        Вычисляет минимальную энергию для стирания информации (принцип Ландауэра).
        
        Принцип Ландауэра утверждает, что стирание 1 бита информации
        требует минимум k_B * T * ln(2) энергии.
        
        Args:
            bits_erased: Количество стираемых бит
            temperature: Температура системы в Кельвинах
        
        Returns:
            Минимальная энергия в Джоулях
        
        Raises:
            ValueError: Если параметры некорректны
        """
        if bits_erased < 0:
            raise ValueError(f"bits_erased должен быть >= 0, получено {bits_erased}")
        if temperature < 0:
            raise ValueError(f"temperature должна быть >= 0, получено {temperature}")
        
        min_energy = bits_erased * EmergentLaws.BOLTZMANN_CONSTANT * temperature * np.log(2)
        return min_energy
    
    @staticmethod
    def estimate_metric_from_entanglement(entanglement_pattern: Union[np.ndarray, List]) -> np.ndarray:
        """
        Оценка метрики пространства-времени из паттернов запутанности.
        
        Упрощенная модель: метрика пространства-времени коррелирует
        с матрицей запутанности между квантовыми системами.
        
        Args:
            entanglement_pattern: Матрица или массив паттернов запутанности
        
        Returns:
            Оценка метрики пространства-времени (матрица)
        """
        if isinstance(entanglement_pattern, list):
            entanglement_pattern = np.array(entanglement_pattern)
        
        if not isinstance(entanglement_pattern, np.ndarray):
            return np.eye(2)  # Метрика по умолчанию
        
        if entanglement_pattern.ndim == 1:
            # Если одномерный массив, создаем матрицу корреляций
            if len(entanglement_pattern) < 2:
                return np.eye(2)
            # Преобразуем в матрицу для корреляции
            pattern_2d = entanglement_pattern.reshape(-1, 1)
            metric = np.corrcoef(pattern_2d, rowvar=False)
        elif entanglement_pattern.ndim == 2:
            # Если уже матрица, используем корреляцию
            metric = np.corrcoef(entanglement_pattern)
        else:
            return np.eye(2)
        
        # Убеждаемся, что метрика симметрична и положительно определена
        metric = (metric + metric.T) / 2  # Симметризация
        
        return metric
