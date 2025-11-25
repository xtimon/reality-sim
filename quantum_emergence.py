# examples/quantum_emergence.py

import sys
import os
import json
import csv
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import functools
import platform

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from reality_sim import QuantumFabric, EmergentLaws, Observer, HUMAN_OBSERVER, LIGO_OBSERVER, ELECTRON_OBSERVER

# Количество доступных ядер CPU
# На Windows используем все ядра, на Unix оставляем одно свободным
if platform.system() == 'Windows':
    NUM_WORKERS = cpu_count()
else:
    NUM_WORKERS = max(1, cpu_count() - 1)  # Оставляем одно ядро свободным

# Максимальное количество кубитов для полных симуляций
# Для complex128: 2^30 * 16 байт ≈ 16 GB, 2^35 ≈ 512 GB (слишком много)
MAX_QUBITS_FULL_SIMULATION = 30  # Безопасный лимит для полных симуляций
MAX_QUBITS_LIMITED = 50  # Для ограниченных симуляций (без полного вектора состояния)

def estimate_memory_requirement(num_qubits: int, dtype_size: int = 16) -> float:
    """
    Оценивает требуемую память для квантовой системы.
    
    Args:
        num_qubits: Количество кубитов
        dtype_size: Размер одного элемента в байтах (complex128 = 16)
    
    Returns:
        Требуемая память в байтах
    """
    return (2 ** num_qubits) * dtype_size

def check_memory_safety(num_qubits: int, max_qubits: int = MAX_QUBITS_FULL_SIMULATION) -> bool:
    """
    Проверяет, безопасно ли создавать систему с данным количеством кубитов.
    
    Args:
        num_qubits: Количество кубитов
        max_qubits: Максимальное безопасное количество кубитов
    
    Returns:
        True если безопасно, False иначе
    """
    return num_qubits <= max_qubits

# Глобальная переменная для хранения всех данных
simulation_data = {
    'timestamp': datetime.now().isoformat(),
    'quantum_systems': [],
    'decoherence_time_series': [],
    'particle_creation': [],
    'measurement_statistics': [],
    'parameter_sweeps': []
}

def save_data_to_json(filename: str = 'simulation_data.json'):
    """Сохраняет все собранные данные в JSON файл"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(simulation_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ Данные сохранены в {filename}")

def save_decoherence_to_csv(filename: str = 'decoherence_data.csv'):
    """Сохраняет временные ряды декогеренции в CSV"""
    if not simulation_data['decoherence_time_series']:
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['observer', 'time', 'coherence', 'decoherence_power'])
        for entry in simulation_data['decoherence_time_series']:
            writer.writerow([
                entry['observer'],
                entry['time'],
                entry['coherence'],
                entry['decoherence_power']
            ])
    print(f"✓ Данные декогеренции сохранены в {filename}")

# Вспомогательные функции для параллелизации
def _simulate_basic_system(config):
    """Вспомогательная функция для параллельного выполнения базовой симуляции"""
    try:
        from reality_sim import QuantumFabric
        
        num_qubits = config['num_qubits']
        if not check_memory_safety(num_qubits):
            mem_gb = estimate_memory_requirement(num_qubits) / (1024**3)
            return {
                'error': f'Слишком много кубитов ({num_qubits}). Требуется ~{mem_gb:.1f} GB памяти.',
                'num_qubits': num_qubits,
                'skipped': True
            }
        
        # Используем GPU для больших систем (если доступен)
        use_gpu = config.get('use_gpu', None) if num_qubits >= 15 else False
        system = QuantumFabric(num_qubits=num_qubits, 
                              entanglement_strength=config['entanglement_strength'],
                              use_gpu=use_gpu)
    initial_info = {
        'num_qubits': system.n,
        'entanglement_strength': system.entanglement_strength,
        'initial_coherence': float(system.get_coherence()),
        'initial_entanglement': float(system.get_entanglement_entropy())
    }
    
    pairs = [(i, i+1) for i in range(system.n - 1)]
    system.apply_entanglement_operator(pairs)
    
    after_entanglement_info = {
        'coherence': float(system.get_coherence()),
        'entanglement_entropy': float(system.get_entanglement_entropy()),
        'entanglement_pairs': pairs
    }
    
    all_stats = {}
    for i in range(system.n):
        stats = system.collect_measurement_statistics(i, 10000)
        all_stats[f'qubit_{i}'] = {
            'count_0': stats['count_0'],
            'count_1': stats['count_1'],
            'prob_0': float(stats['prob_0']),
            'prob_1': float(stats['prob_1']),
            'num_measurements': stats['num_measurements']
        }
    
    qubit_probs = {}
    for i in range(system.n):
        prob_0, prob_1 = system.get_qubit_probabilities(i)
        qubit_probs[f'qubit_{i}'] = {'prob_0': float(prob_0), 'prob_1': float(prob_1)}
    
        return {
            **initial_info,
            **after_entanglement_info,
            'measurement_stats': all_stats,
            'qubit_probabilities': qubit_probs
        }
    except Exception as e:
        # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
        error_msg = str(e) if e else repr(e)
        return {
            'error': error_msg,
            'num_qubits': config.get('num_qubits', 'unknown'),
            'skipped': True
        }

def _simulate_particle_creation(args):
    """Вспомогательная функция для параллельного выполнения симуляции рождения частиц"""
    try:
        from reality_sim import EmergentLaws
        vacuum_energy, time_steps = args
        
        particles = EmergentLaws.simulate_particle_creation(
            vacuum_energy=vacuum_energy, 
            time_steps=time_steps
        )
        
        creation_times = [p[0]['created_at'] for p in particles]
        
        return {
            'vacuum_energy': float(vacuum_energy),
            'time_steps': time_steps,
            'total_pairs': len(particles),
            'creation_times': creation_times,
            'mean_creation_time': float(np.mean(creation_times)) if creation_times else 0.0,
            'std_creation_time': float(np.std(creation_times)) if creation_times else 0.0,
            'first_3_pairs': [
                {
                    'particle': p[0],
                    'antiparticle': p[1]
                } for p in particles[:3]
            ]
        }
    except Exception as e:
        # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
        error_msg = str(e) if e else repr(e)
        return {
            'error': error_msg,
            'vacuum_energy': args[0] if args else 'unknown',
            'time_steps': args[1] if args and len(args) > 1 else 'unknown',
            'skipped': True
        }

def _simulate_entanglement_config(config):
    """Вспомогательная функция для параллельного выполнения анализа запутанности"""
    try:
        from reality_sim import QuantumFabric
        
        num_qubits = config['num_qubits']
        if not check_memory_safety(num_qubits):
            mem_gb = estimate_memory_requirement(num_qubits) / (1024**3)
            return {
                'error': f'Слишком много кубитов ({num_qubits}). Требуется ~{mem_gb:.1f} GB памяти.',
                'num_qubits': num_qubits,
                'skipped': True
            }
        
        # Используем GPU для больших систем (если доступен)
        use_gpu = None if num_qubits >= 15 else False
        system = QuantumFabric(num_qubits=num_qubits, 
                              entanglement_strength=config['strength'],
                              use_gpu=use_gpu)
        initial_ent = system.get_entanglement_entropy()
        
        system.apply_entanglement_operator(config['pairs'])
        final_ent = system.get_entanglement_entropy()
        
        qubit_probs = {
            i: system.get_qubit_probabilities(i) 
            for i in range(system.n)
        }
        
        return {
            'num_qubits': config['num_qubits'],
            'entanglement_pairs': config['pairs'],
            'entanglement_strength': config['strength'],
            'initial_entanglement': float(initial_ent),
            'final_entanglement': float(final_ent),
            'coherence': float(system.get_coherence()),
            'qubit_probabilities': {
                str(k): {'prob_0': float(v[0]), 'prob_1': float(v[1])}
                for k, v in qubit_probs.items()
            }
        }
    except Exception as e:
        # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
        error_msg = str(e) if e else repr(e)
        return {
            'error': error_msg,
            'num_qubits': config.get('num_qubits', 'unknown'),
            'skipped': True
        }

def _simulate_measurement_config(config):
    """Вспомогательная функция для параллельного выполнения статистики измерений"""
    try:
        from reality_sim import QuantumFabric
        
        num_qubits = config['num_qubits']
        if not check_memory_safety(num_qubits):
            mem_gb = estimate_memory_requirement(num_qubits) / (1024**3)
            return {
                'error': f'Слишком много кубитов ({num_qubits}). Требуется ~{mem_gb:.1f} GB памяти.',
                'num_qubits': num_qubits,
                'skipped': True
            }
        
        # Используем GPU для больших систем (если доступен)
        use_gpu = None if num_qubits >= 15 else False
        system = QuantumFabric(num_qubits=num_qubits, use_gpu=use_gpu)
        system.apply_entanglement_operator(config['pairs'])
        
        stats_per_qubit = {}
        for i in range(system.n):
            stats = system.collect_measurement_statistics(i, config['num_measurements'])
            stats_per_qubit[f'qubit_{i}'] = {
                'count_0': stats['count_0'],
                'count_1': stats['count_1'],
                'prob_0': float(stats['prob_0']),
                'prob_1': float(stats['prob_1']),
                'num_measurements': stats['num_measurements']
            }
        
        return {
            'num_qubits': config['num_qubits'],
            'entanglement_pairs': config['pairs'],
            'num_measurements': config['num_measurements'],
            'measurement_stats': stats_per_qubit,
            'coherence': float(system.get_coherence()),
            'entanglement_entropy': float(system.get_entanglement_entropy())
        }
    except Exception as e:
        # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
        error_msg = str(e) if e else repr(e)
        return {
            'error': error_msg,
            'num_qubits': config.get('num_qubits', 'unknown'),
            'skipped': True
        }

def _simulate_large_system(num_qubits):
    """Вспомогательная функция для параллельного выполнения больших систем"""
    try:
        from reality_sim import QuantumFabric
        
        if not check_memory_safety(num_qubits):
            mem_gb = estimate_memory_requirement(num_qubits) / (1024**3)
            return {
                'error': f'Слишком много кубитов ({num_qubits}). Требуется ~{mem_gb:.1f} GB памяти.',
                'num_qubits': num_qubits,
                'skipped': True
            }
        
        # Для больших систем используем GPU если доступен
        system = QuantumFabric(num_qubits=num_qubits, entanglement_strength=1.0, use_gpu=None)
        max_pairs = min(30, num_qubits - 1)
        pairs = [(i, i+1) for i in range(max_pairs)]
        system.apply_entanglement_operator(pairs)
        
        return {
            'num_qubits': num_qubits,
            'entanglement_pairs': len(pairs),
            'entanglement_entropy': float(system.get_entanglement_entropy()),
            'coherence': float(system.get_coherence())
        }
    except Exception as e:
        # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
        error_msg = str(e) if e else repr(e)
        return {
            'error': error_msg,
            'num_qubits': num_qubits,
            'skipped': True
        }

def _simulate_strength_sweep(strength):
    """Вспомогательная функция для sweep по силе запутанности"""
    try:
        from reality_sim import QuantumFabric
        
        system = QuantumFabric(num_qubits=2, entanglement_strength=strength)
        system.apply_entanglement_operator([(0, 1)])
        
        return {
            'entanglement_strength': float(strength),
            'entanglement_entropy': float(system.get_entanglement_entropy()),
            'coherence': float(system.get_coherence()),
            'prob_0_qubit0': float(system.get_qubit_probabilities(0)[0]),
            'prob_1_qubit0': float(system.get_qubit_probabilities(0)[1]),
            'prob_0_qubit1': float(system.get_qubit_probabilities(1)[0]),
            'prob_1_qubit1': float(system.get_qubit_probabilities(1)[1])
        }
    except Exception as e:
        # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
        error_msg = str(e) if e else repr(e)
        return {
            'error': error_msg,
            'entanglement_strength': strength,
            'skipped': True
        }

def _simulate_qubit_count_sweep(num_qubits):
    """Вспомогательная функция для sweep по количеству кубитов"""
    try:
        from reality_sim import QuantumFabric
        
        if not check_memory_safety(num_qubits):
            mem_gb = estimate_memory_requirement(num_qubits) / (1024**3)
            return {
                'error': f'Слишком много кубитов ({num_qubits}). Требуется ~{mem_gb:.1f} GB памяти.',
                'num_qubits': num_qubits,
                'skipped': True
            }
        
        # Используем GPU для больших систем
        use_gpu = None if num_qubits >= 15 else False
        system = QuantumFabric(num_qubits=num_qubits, entanglement_strength=1.0, use_gpu=use_gpu)
        pairs = [(i, i+1) for i in range(num_qubits - 1)]
        system.apply_entanglement_operator(pairs)
        
        return {
            'num_qubits': num_qubits,
            'entanglement_pairs': len(pairs),
            'entanglement_entropy': float(system.get_entanglement_entropy()),
            'coherence': float(system.get_coherence())
        }
    except Exception as e:
        # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
        error_msg = str(e) if e else repr(e)
        return {
            'error': error_msg,
            'num_qubits': num_qubits,
            'skipped': True
        }

def _simulate_multi_qubit_strength(args):
    """Вспомогательная функция для sweep силы запутанности для разных систем"""
    try:
        from reality_sim import QuantumFabric
        num_qubits, strength = args
        
        if not check_memory_safety(num_qubits):
            mem_gb = estimate_memory_requirement(num_qubits) / (1024**3)
            return {
                'error': f'Слишком много кубитов ({num_qubits}). Требуется ~{mem_gb:.1f} GB памяти.',
                'num_qubits': num_qubits,
                'skipped': True
            }
        
        # Используем GPU для больших систем
        use_gpu = None if num_qubits >= 15 else False
        system = QuantumFabric(num_qubits=num_qubits, entanglement_strength=strength, use_gpu=use_gpu)
        pairs = [(i, i+1) for i in range(min(20, num_qubits - 1))]
        system.apply_entanglement_operator(pairs)
        
        return {
            'num_qubits': num_qubits,
            'entanglement_strength': float(strength),
            'entanglement_entropy': float(system.get_entanglement_entropy()),
            'coherence': float(system.get_coherence())
        }
    except Exception as e:
        # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
        error_msg = str(e) if e else repr(e)
        return {
            'error': error_msg,
            'num_qubits': args[0] if args else 'unknown',
            'skipped': True
        }

def demo_basic_quantum_system():
    """Демонстрация базовой квантовой системы с расширенным анализом"""
    print("=== Базовая квантовая система ===")
    
    # Значительно расширенные множественные базовые системы с разными параметрами
    basic_systems_configs = []
    # Генерируем конфигурации для разных количеств кубитов и сил запутанности
    # Для малых систем (2-10) - все силы запутанности
    for num_qubits in range(2, 11):
        for strength in [0.25, 0.5, 0.75, 1.0]:
            basic_systems_configs.append({
                'num_qubits': num_qubits,
                'entanglement_strength': strength
            })
    # Для средних систем (11-30) - только полная запутанность
    for num_qubits in range(11, 31, 2):  # Каждые 2 кубита
        basic_systems_configs.append({
            'num_qubits': num_qubits,
            'entanglement_strength': 1.0
        })
    # Для больших систем (31-30) - только полная запутанность, реже
    # Ограничиваем до MAX_QUBITS_FULL_SIMULATION для безопасности памяти
    for num_qubits in range(31, min(MAX_QUBITS_FULL_SIMULATION + 1, 51), 5):  # Каждые 5 кубитов
        if num_qubits <= MAX_QUBITS_FULL_SIMULATION:
            basic_systems_configs.append({
                'num_qubits': num_qubits,
                'entanglement_strength': 1.0
            })
    
    print(f"Используется {NUM_WORKERS} параллельных процессов для {len(basic_systems_configs)} конфигураций...")
    
    # Параллельное выполнение
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_simulate_basic_system, config): idx 
                   for idx, config in enumerate(basic_systems_configs)}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == len(basic_systems_configs):
                print(f"  Обработано {completed}/{len(basic_systems_configs)} систем...")
            try:
                result = future.result()
                if isinstance(result, dict) and result.get('skipped'):
                    print(f"  ⚠ Пропущено: {result.get('error', 'Недостаточно памяти')}")
                else:
                    results.append(result)
            except MemoryError as e:
                config_idx = futures[future]
                print(f"  ⚠ Ошибка памяти при обработке конфигурации {config_idx}: {str(e)}")
            except Exception as e:
                config_idx = futures[future]
                # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
                error_msg = str(e) if e else repr(e)
                print(f"  ⚠ Ошибка при обработке конфигурации {config_idx}: {error_msg}")
    
    # Показываем детали для первых 3 систем
    if results:
        print("\n--- Детали первых 3 систем ---")
        for idx, result in enumerate(results[:3]):
            if 'num_qubits' in result and 'initial_entanglement' in result:
                print(f"\nСистема {idx + 1}: {result['num_qubits']} кубитов")
                print(f"  Запутанность: {result['initial_entanglement']:.4f} → {result['entanglement_entropy']:.4f}")
                print(f"  Когерентность: {result['coherence']:.6f}")
    
    simulation_data['quantum_systems'].extend(results)
    
    return results[0] if results else None

def demo_particle_creation():
    """Демонстрация рождения частиц из вакуума с расширенной статистикой"""
    print("\n=== Симуляция рождения пар частица-античастица ===")
    
    # Значительно расширенные множественные симуляции с разными параметрами
    vacuum_energies = np.linspace(0.01, 0.6, 30)  # 30 значений энергии
    time_steps_range = list(range(20, 501, 20))  # От 20 до 500 с шагом 20 = 25 значений
    
    # Создаем список всех комбинаций параметров
    param_combinations = [(vacuum_energy, time_steps) 
                          for vacuum_energy in vacuum_energies 
                          for time_steps in time_steps_range]
    
    print(f"Используется {NUM_WORKERS} параллельных процессов для {len(param_combinations)} симуляций...")
    
    # Параллельное выполнение
    all_particles_data = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_simulate_particle_creation, args): args 
                   for args in param_combinations}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                print(f"  Выполнено {completed}/{len(param_combinations)} симуляций...")
            try:
                result = future.result()
                all_particles_data.append(result)
            except Exception as e:
                args = futures[future]
                # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
                error_msg = str(e) if e else repr(e)
                print(f"  Ошибка при обработке (energy={args[0]}, steps={args[1]}): {error_msg}")
    
    print(f"\nВсего выполнено {len(all_particles_data)} симуляций рождения частиц")
    print(f"Энергии вакуума: {len(vacuum_energies)} значений")
    print(f"Временные шаги: {len(time_steps_range)} значений")
    
    simulation_data['particle_creation'] = all_particles_data
    
    # Показываем детали для нескольких случаев
    test_cases = [
        (0.1, 50), (0.2, 100), (0.3, 200)
    ]
    for vacuum_energy, time_steps in test_cases:
        particles = EmergentLaws.simulate_particle_creation(vacuum_energy=vacuum_energy, time_steps=time_steps)
        print(f"\nДетали для энергии={vacuum_energy}, шагов={time_steps}:")
        print(f"Создано {len(particles)} пар частица-античастица")
        for i, (p, ap) in enumerate(particles[:3]):  # Покажем первые 3
            print(f"  Пара {i+1}: {p['type']} (+{p['charge']}) + {ap['type']} ({ap['charge']}) в t={p['created_at']}")

def demo_observer_effect():
    """Демонстрация разной силы декогеренции у наблюдателей с временными рядами"""
    print("\n=== Сравнение силы наблюдателей ===")
    
    observers = [HUMAN_OBSERVER, LIGO_OBSERVER, ELECTRON_OBSERVER]
    names = ["Человек", "LIGO", "Электрон"]
    
    # Добавляем множество дополнительных наблюдателей
    additional_observers = [
        Observer(mass=1e-3, temperature=300, complexity=1e5),  # Мелкий объект
        Observer(mass=1e6, temperature=100, complexity=1e12),  # Большой объект
        Observer(mass=70, temperature=77, complexity=1e15),   # Человек при низкой температуре
        Observer(mass=1e-6, temperature=300, complexity=1e3),  # Очень мелкий объект
        Observer(mass=1e9, temperature=50, complexity=1e15),  # Очень большой объект
        Observer(mass=1, temperature=4, complexity=1e8),  # Объект при сверхнизкой температуре
        Observer(mass=1000, temperature=1000, complexity=1e10),  # Горячий объект
        Observer(mass=1e-9, temperature=2.7, complexity=1),  # Космический микроб
        Observer(mass=1e12, temperature=300, complexity=1e18),  # Планетарный масштаб
        Observer(mass=70, temperature=310, complexity=1e16),  # Человек при повышенной температуре
    ]
    additional_names = [
        "Мелкий объект", "Большой объект", "Человек (77K)",
        "Очень мелкий объект", "Очень большой объект", "Объект (4K)",
        "Горячий объект", "Космический микроб", "Планетарный масштаб",
        "Человек (310K)"
    ]
    
    all_observers = observers + additional_observers
    all_names = names + additional_names
    
    test_system = 1.0  # Идеально когерентное состояние
    
    # Мгновенные измерения
    print("--- Мгновенные измерения ---")
    for name, observer in zip(all_names, all_observers):
        result = observer.observe_system(test_system, observation_time=1.0)
        coherence_left = abs(result)
        
        print(f"{name}:")
        print(f"  Сила декогеренции Γ = {observer.decoherence_power:.2e}")
        print(f"  Сохраненная когерентность = {coherence_left:.2e}")
    
    # Значительно расширенные временные ряды декогеренции
    print("\n--- Временные ряды декогеренции ---")
    time_points = np.linspace(0, 20, 1000)  # 1000 точек от 0 до 20 секунд
    
    for name, observer in zip(all_names, all_observers):
        coherence_series = []
        for t in time_points:
            result = observer.observe_system(test_system, observation_time=t)
            coherence = abs(result)
            coherence_series.append(coherence)
            
            simulation_data['decoherence_time_series'].append({
                'observer': name,
                'time': float(t),
                'coherence': float(coherence),
                'decoherence_power': float(observer.decoherence_power)
            })
        
        half_coherence_idx = np.argmin(np.abs(np.array(coherence_series) - 0.5))
        half_coherence_time = time_points[half_coherence_idx] if half_coherence_idx < len(time_points) else 20.0
        print(f"{name}: когерентность упала с 1.0 до {coherence_series[-1]:.6f} за 20 сек")
        print(f"  Половина когерентности при t ≈ {half_coherence_time:.2f} сек")

def demo_landauer_limit():
    """Демонстрация принципа Ландауэра с анализом зависимостей"""
    print("\n=== Принцип Ландауэра ===")
    
    # Базовые примеры
    print("--- Базовые примеры ---")
    for bits in [1, 8, 1024, 1024*1024]:
        energy = EmergentLaws.landauer_principle(bits, temperature=300)
        print(f"Стирание {bits} бит при 300K: {energy:.2e} Дж")
    
    # Расширенная зависимость от температуры
    print("\n--- Зависимость от температуры (1024 бит) ---")
    temperatures = np.logspace(0, 3, 100)  # От 1 до 1000 K, 100 точек в логарифмической шкале
    temp_data = []
    for temp in temperatures:
        energy = EmergentLaws.landauer_principle(1024, temperature=temp)
        temp_data.append({'temperature': float(temp), 'energy': float(energy)})
    
    print(f"Выполнено {len(temperatures)} симуляций для разных температур")
    print("Примеры (выборочно):")
    for i in [0, 25, 50, 75, 99]:
        print(f"  T = {temp_data[i]['temperature']:.2f} K: {temp_data[i]['energy']:.2e} Дж")
    
    # Расширенная зависимость от количества бит
    print("\n--- Зависимость от количества бит (300K) ---")
    bit_counts = np.logspace(0, 8, 200)  # От 1 до 100M бит, 200 точек в логарифмической шкале
    bits_data = []
    for bits in bit_counts:
        energy = EmergentLaws.landauer_principle(int(bits), temperature=300)
        bits_data.append({'bits': int(bits), 'energy': float(energy)})
    
    print(f"Выполнено {len(bit_counts)} симуляций для разных количеств бит")
    print("Примеры (выборочно):")
    for i in [0, 50, 100, 150, 199]:
        print(f"  {bits_data[i]['bits']} бит: {bits_data[i]['energy']:.2e} Дж")
    
    # Сравнение с повседневными процессами
    print("\n--- Сравнение с реальными системами ---")
    systems = {
        'Человеческий мозг': 1e15,
        'Современный компьютер (RAM)': 1e12,
        'Квантовый компьютер (1000 кубитов)': 1000
    }
    
    for name, bits in systems.items():
        energy = EmergentLaws.landauer_principle(bits, 300)
        print(f"{name} ({bits:.0e} бит): {energy:.2e} Дж/с")
    
    simulation_data['parameter_sweeps'].append({
        'type': 'landauer_temperature',
        'data': temp_data
    })
    simulation_data['parameter_sweeps'].append({
        'type': 'landauer_bits',
        'data': bits_data
    })

def demo_entanglement_analysis():
    """Анализ запутанности для разных конфигураций"""
    print("\n=== Анализ запутанности ===")
    
    # Генерируем значительно больше конфигураций
    configurations = []
    
    # 2 кубита с разной силой запутанности
    for strength in np.linspace(0.1, 1.0, 10):
        configurations.append({'num_qubits': 2, 'pairs': [(0, 1)], 'strength': strength})
    
    # 3 кубита - все возможные комбинации пар
    pairs_3 = [
        [(0, 1)],
        [(1, 2)],
        [(0, 2)],
        [(0, 1), (1, 2)],
        [(0, 1), (0, 2)],
        [(0, 1), (1, 2), (0, 2)],
    ]
    for pairs in pairs_3:
        for strength in [0.5, 0.75, 1.0]:
            configurations.append({'num_qubits': 3, 'pairs': pairs, 'strength': strength})
    
    # 4 кубита - различные паттерны
    pairs_4 = [
        [(0, 1)],
        [(0, 1), (2, 3)],
        [(0, 1), (1, 2)],
        [(0, 1), (1, 2), (2, 3)],
        [(0, 1), (0, 2), (0, 3)],
        [(0, 1), (1, 2), (2, 3), (3, 0)],
        [(0, 1), (1, 2), (2, 3), (0, 2)],
    ]
    for pairs in pairs_4:
        for strength in [0.5, 0.75, 1.0]:
            configurations.append({'num_qubits': 4, 'pairs': pairs, 'strength': strength})
    
    # 5 кубитов
    pairs_5 = [
        [(0, 1), (2, 3)],
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        [(0, 1), (0, 2), (0, 3), (0, 4)],
        [(0, 1), (1, 2), (3, 4)],
    ]
    for pairs in pairs_5:
        configurations.append({'num_qubits': 5, 'pairs': pairs, 'strength': 1.0})
    
    # 6 кубитов
    pairs_6 = [
        [(0, 1), (2, 3), (4, 5)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        [(0, 1), (2, 3), (4, 5), (0, 3)],
    ]
    for pairs in pairs_6:
        configurations.append({'num_qubits': 6, 'pairs': pairs, 'strength': 1.0})
    
    # 7 кубитов
    pairs_7 = [
        [(0, 1), (2, 3), (4, 5)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
    ]
    for pairs in pairs_7:
        configurations.append({'num_qubits': 7, 'pairs': pairs, 'strength': 1.0})
    
    # 8 кубитов
    pairs_8 = [
        [(0, 1), (2, 3), (4, 5), (6, 7)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
    ]
    for pairs in pairs_8:
        configurations.append({'num_qubits': 8, 'pairs': pairs, 'strength': 1.0})
    
    # Системы от 9 до MAX_QUBITS_FULL_SIMULATION кубитов - используем простые паттерны
    max_qubits = min(MAX_QUBITS_FULL_SIMULATION, 50)
    for num_qubits in range(9, max_qubits + 1):
        if num_qubits <= 20:
            # Для систем до 20 кубитов - несколько паттернов
            patterns = [
                [(i, i+1) for i in range(0, num_qubits-1, 2)],  # Четные пары
                [(i, i+1) for i in range(num_qubits - 1)],  # Все соседние
            ]
            for pairs in patterns:
                configurations.append({'num_qubits': num_qubits, 'pairs': pairs, 'strength': 1.0})
        else:
            # Для больших систем (21-MAX_QUBITS_FULL_SIMULATION) - только один простой паттерн
            pairs = [(i, i+1) for i in range(0, min(20, num_qubits-1))]  # Первые 20 пар
            configurations.append({'num_qubits': num_qubits, 'pairs': pairs, 'strength': 1.0})
    
    print(f"Используется {NUM_WORKERS} параллельных процессов для {len(configurations)} конфигураций...")
    
    # Параллельное выполнение
    entanglement_data = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_simulate_entanglement_config, config): idx 
                   for idx, config in enumerate(configurations)}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0 or completed == len(configurations):
                print(f"  Обработано {completed}/{len(configurations)} конфигураций...")
            try:
                result = future.result()
                if isinstance(result, dict) and result.get('skipped'):
                    print(f"  ⚠ Пропущено: {result.get('error', 'Недостаточно памяти')}")
                else:
                    entanglement_data.append(result)
            except MemoryError as e:
                config_idx = futures[future]
                print(f"  ⚠ Ошибка памяти при обработке конфигурации {config_idx}: {str(e)}")
            except Exception as e:
                config_idx = futures[future]
                # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
                error_msg = str(e) if e else repr(e)
                print(f"  ⚠ Ошибка при обработке конфигурации {config_idx}: {error_msg}")
    
    print(f"\nПоказаны результаты для первых 5 конфигураций:")
    for config, data in zip(configurations[:5], entanglement_data[:5]):
        print(f"\n{config['num_qubits']} кубитов, пары: {config['pairs']}, сила: {config['strength']}")
        print(f"  Запутанность: {data['initial_entanglement']:.4f} → {data['final_entanglement']:.4f}")
        print(f"  Когерентность: {data['coherence']:.6f}")
    
    simulation_data['quantum_systems'].extend(entanglement_data)

def demo_parameter_sweep():
    """Анализ зависимости от параметров запутанности"""
    print("\n=== Анализ зависимости от силы запутанности ===")
    
    # Значительно расширенный sweep по силе запутанности
    entanglement_strengths = np.linspace(0.01, 1.0, 200)  # 200 точек
    
    print(f"Используется {NUM_WORKERS} параллельных процессов для {len(entanglement_strengths)} значений...")
    
    # Параллельное выполнение
    sweep_data = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_simulate_strength_sweep, strength): strength 
                   for strength in entanglement_strengths}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0:
                print(f"  Обработано {completed}/{len(entanglement_strengths)} значений...")
            try:
                result = future.result()
                sweep_data.append(result)
            except Exception as e:
                strength = futures[future]
                # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
                error_msg = str(e) if e else repr(e)
                print(f"  Ошибка при обработке strength={strength}: {error_msg}")
    
    # Сортируем по силе запутанности
    sweep_data.sort(key=lambda x: x['entanglement_strength'])
    
    print("Сила запутанности → Энтропия запутанности (выборочно):")
    for entry in sweep_data[::10]:  # Показываем каждую 10-ю точку
        print(f"  {entry['entanglement_strength']:.2f} → {entry['entanglement_entropy']:.4f}")
    
    simulation_data['parameter_sweeps'].append({
        'type': 'entanglement_strength',
        'data': sweep_data
    })
    
    # Дополнительный sweep по количеству кубитов
    print("\n=== Анализ зависимости от количества кубитов ===")
    # От 2 до MAX_QUBITS_FULL_SIMULATION кубитов, с разной частотой для больших систем
    qubit_counts = list(range(2, 21))  # От 2 до 20 - все
    # Ограничиваем до безопасного лимита
    max_safe = min(MAX_QUBITS_FULL_SIMULATION, 50)
    if max_safe > 20:
        qubit_counts.extend(range(25, max_safe + 1, 5))  # От 25 до max_safe - каждые 5
    
    print(f"Используется {NUM_WORKERS} параллельных процессов для {len(qubit_counts)} значений...")
    
    # Параллельное выполнение
    qubit_sweep_data = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_simulate_qubit_count_sweep, num_qubits): num_qubits 
                   for num_qubits in qubit_counts}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
                if isinstance(result, dict) and result.get('skipped'):
                    print(f"  ⚠ Пропущено: {result.get('error', 'Недостаточно памяти')}")
                else:
                    qubit_sweep_data.append(result)
            except MemoryError as e:
                num_qubits = futures[future]
                print(f"  ⚠ Ошибка памяти при обработке {num_qubits} кубитов: {str(e)}")
            except Exception as e:
                num_qubits = futures[future]
                # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
                error_msg = str(e) if e else repr(e)
                print(f"  ⚠ Ошибка при обработке {num_qubits} кубитов: {error_msg}")
    
    # Сортируем по количеству кубитов
    qubit_sweep_data.sort(key=lambda x: x['num_qubits'])
    
    print("Количество кубитов → Энтропия запутанности:")
    for entry in qubit_sweep_data:
        print(f"  {entry['num_qubits']} кубитов ({entry['entanglement_pairs']} пар) → {entry['entanglement_entropy']:.4f}")
    
    simulation_data['parameter_sweeps'].append({
        'type': 'qubit_count',
        'data': qubit_sweep_data
    })
    
    # Дополнительный sweep по силе запутанности для разных количеств кубитов
    print("\n=== Sweep силы запутанности для разных систем ===")
    
    # Создаем список всех комбинаций
    multi_qubit_params = []
    # Для малых систем - детальный sweep
    for num_qubits in [2, 3, 4, 5]:
        for strength in np.linspace(0.1, 1.0, 50):
            multi_qubit_params.append((num_qubits, strength))
    # Для больших систем - только несколько значений силы (ограничиваем до безопасного лимита)
    large_systems = [10, 20, 30]
    if MAX_QUBITS_FULL_SIMULATION >= 40:
        large_systems.append(40)
    if MAX_QUBITS_FULL_SIMULATION >= 50:
        large_systems.append(50)
    
    for num_qubits in large_systems:
        if num_qubits <= MAX_QUBITS_FULL_SIMULATION:
            for strength in [0.25, 0.5, 0.75, 1.0]:
                multi_qubit_params.append((num_qubits, strength))
    
    print(f"Используется {NUM_WORKERS} параллельных процессов для {len(multi_qubit_params)} комбинаций...")
    
    # Параллельное выполнение
    multi_qubit_sweep = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_simulate_multi_qubit_strength, args): args 
                   for args in multi_qubit_params}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0:
                print(f"  Обработано {completed}/{len(multi_qubit_params)} комбинаций...")
            try:
                result = future.result()
                if isinstance(result, dict) and result.get('skipped'):
                    print(f"  ⚠ Пропущено: {result.get('error', 'Недостаточно памяти')}")
                else:
                    multi_qubit_sweep.append(result)
            except MemoryError as e:
                args = futures[future]
                print(f"  ⚠ Ошибка памяти при обработке (qubits={args[0]}, strength={args[1]}): {str(e)}")
            except Exception as e:
                args = futures[future]
                # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
                error_msg = str(e) if e else repr(e)
                print(f"  ⚠ Ошибка при обработке (qubits={args[0]}, strength={args[1]}): {error_msg}")
    
    simulation_data['parameter_sweeps'].append({
        'type': 'multi_qubit_strength',
        'data': multi_qubit_sweep
    })
    print(f"Выполнено {len(multi_qubit_sweep)} симуляций для разных систем")

def demo_multi_measurement_statistics():
    """Расширенная статистика измерений для разных систем"""
    print("\n=== Расширенная статистика измерений ===")
    
    # Значительно расширенные конфигурации измерений
    measurement_configs = []
    # Генерируем конфигурации для разных систем
    # Малые системы (2-10) - детальные конфигурации
    for num_qubits in range(2, 11):
        if num_qubits == 2:
            pairs_list = [[(0, 1)]]
        elif num_qubits == 3:
            pairs_list = [[(0, 1)], [(0, 1), (1, 2)]]
        elif num_qubits <= 6:
            # Для 4-6 кубитов - несколько паттернов
            pairs_list = [
                [(i, i+1) for i in range(0, num_qubits-1, 2)],  # Четные пары
                [(i, i+1) for i in range(num_qubits - 1)],  # Все соседние
            ]
        else:
            # Для 7-10 кубитов - один паттерн
            pairs_list = [[(i, i+1) for i in range(num_qubits - 1)]]
        
        for pairs in pairs_list:
            for num_measurements in [10000, 20000, 50000]:
                measurement_configs.append({
                    'num_qubits': num_qubits,
                    'num_measurements': num_measurements,
                    'pairs': pairs
                })
    
    # Средние системы (11-30) - меньше конфигураций
    for num_qubits in range(15, 31, 5):  # 15, 20, 25, 30
        pairs = [(i, i+1) for i in range(min(20, num_qubits - 1))]  # Ограничиваем количество пар
        for num_measurements in [10000, 50000]:
            measurement_configs.append({
                'num_qubits': num_qubits,
                'num_measurements': num_measurements,
                'pairs': pairs
            })
    
    # Большие системы (31-MAX_QUBITS_FULL_SIMULATION) - минимальные конфигурации
    max_safe = min(MAX_QUBITS_FULL_SIMULATION, 50)
    if max_safe >= 35:
        for num_qubits in range(35, max_safe + 1, 5):  # 35, 40, 45, 50 (если безопасно)
            if num_qubits <= MAX_QUBITS_FULL_SIMULATION:
                pairs = [(i, i+1) for i in range(min(20, num_qubits - 1))]  # Ограничиваем количество пар
                measurement_configs.append({
                    'num_qubits': num_qubits,
                    'num_measurements': 10000,  # Меньше измерений для больших систем
                    'pairs': pairs
                })
    
    print(f"Используется {NUM_WORKERS} параллельных процессов для {len(measurement_configs)} конфигураций...")
    
    # Параллельное выполнение
    measurement_data = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_simulate_measurement_config, config): idx 
                   for idx, config in enumerate(measurement_configs)}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == len(measurement_configs):
                print(f"  Обработано {completed}/{len(measurement_configs)} конфигураций...")
            try:
                result = future.result()
                if isinstance(result, dict) and result.get('skipped'):
                    print(f"  ⚠ Пропущено: {result.get('error', 'Недостаточно памяти')}")
                else:
                    measurement_data.append(result)
                    
                    if completed <= 5:  # Показываем детали только для первых 5
                        print(f"\n{result['num_qubits']} кубитов, {result['num_measurements']} измерений:")
                        for i in range(result['num_qubits']):
                            stats = result['measurement_stats'][f'qubit_{i}']
                            print(f"  Кубит {i}: P(|0>)={stats['prob_0']:.4f}, P(|1>)={stats['prob_1']:.4f}")
            except MemoryError as e:
                config_idx = futures[future]
                print(f"  ⚠ Ошибка памяти при обработке конфигурации {config_idx}: {str(e)}")
            except Exception as e:
                config_idx = futures[future]
                # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
                error_msg = str(e) if e else repr(e)
                print(f"  ⚠ Ошибка при обработке конфигурации {config_idx}: {error_msg}")
    
    simulation_data['measurement_statistics'].extend(measurement_data)

if __name__ == "__main__":
    print("=" * 60)
    print("РАСШИРЕННАЯ СИМУЛЯЦИЯ КВАНТОВОЙ ЭМЕРДЖЕНТНОСТИ")
    print("=" * 60)
    print(f"⚡ Параллелизация: используется {NUM_WORKERS} ядер CPU")
    
    # Проверка GPU
    try:
        from reality_sim.core.gpu_backend import is_gpu_available, get_device_info
        gpu_info = get_device_info()
        if gpu_info.get('cuda_available'):
            print(f"🎮 GPU: доступен ({gpu_info.get('gpu_name', 'NVIDIA GPU')})")
            print(f"   Системы с ≥15 кубитами будут использовать GPU для ускорения")
        else:
            print(f"💻 GPU: недоступен (используется CPU)")
            print(f"   Для использования GPU установите: pip install cupy-cuda12x")
    except:
        print(f"💻 GPU: недоступен (используется CPU)")
    
    print(f"⚠ Внимание: Максимальное количество кубитов для полных симуляций: {MAX_QUBITS_FULL_SIMULATION}")
    print(f"   Системы с >{MAX_QUBITS_FULL_SIMULATION} кубитами будут пропущены из-за требований памяти.")
    mem_30 = estimate_memory_requirement(30) / (1024**3)
    print(f"   Для {MAX_QUBITS_FULL_SIMULATION} кубитов требуется ~{mem_30:.1f} GB памяти.")
    print("=" * 60)
    
    demo_basic_quantum_system()
    demo_particle_creation()
    demo_observer_effect() 
    demo_landauer_limit()
    demo_entanglement_analysis()
    demo_parameter_sweep()
    demo_multi_measurement_statistics()
    
    # Дополнительные расширенные симуляции
    print("\n" + "=" * 60)
    print("ДОПОЛНИТЕЛЬНЫЕ РАСШИРЕННЫЕ СИМУЛЯЦИИ")
    print("=" * 60)
    
    # Дополнительный анализ запутанности с большим количеством кубитов
    print("\n=== Дополнительный анализ больших систем ===")
    large_systems = []
    # Анализ систем от 10 до MAX_QUBITS_FULL_SIMULATION кубитов
    large_qubit_counts = list(range(10, 21))  # 10-20 все
    max_safe = min(MAX_QUBITS_FULL_SIMULATION, 50)
    if max_safe > 20:
        large_qubit_counts.extend(range(25, max_safe + 1, 5))  # 25-max_safe каждые 5
    
    print(f"Используется {NUM_WORKERS} параллельных процессов для {len(large_qubit_counts)} больших систем...")
    
    # Параллельное выполнение
    large_systems = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_simulate_large_system, num_qubits): num_qubits 
                   for num_qubits in large_qubit_counts}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 5 == 0 or completed == len(large_qubit_counts):
                print(f"  Обработано {completed}/{len(large_qubit_counts)} систем...")
            try:
                result = future.result()
                if isinstance(result, dict) and result.get('skipped'):
                    print(f"  ⚠ Пропущено: {result.get('error', 'Недостаточно памяти')}")
                else:
                    large_systems.append(result)
                    
                    if completed <= 5 or completed % 5 == 0:
                        print(f"{result['num_qubits']} кубитов ({result['entanglement_pairs']} пар): "
                              f"энтропия = {result['entanglement_entropy']:.4f}")
            except MemoryError as e:
                num_qubits = futures[future]
                print(f"  ⚠ Ошибка памяти при обработке системы с {num_qubits} кубитами: {str(e)}")
            except Exception as e:
                num_qubits = futures[future]
                # Преобразуем исключение в строку, чтобы избежать проблем с pickle для PyOpenCL ошибок
                error_msg = str(e) if e else repr(e)
                print(f"  ⚠ Ошибка при обработке системы с {num_qubits} кубитами: {error_msg}")
    
    simulation_data['quantum_systems'].extend(large_systems)
    
    # Сохранение всех данных
    print("\n" + "=" * 60)
    print("СОХРАНЕНИЕ ДАННЫХ")
    print("=" * 60)
    save_data_to_json()
    save_decoherence_to_csv()
    
    print(f"\n✓ Всего собрано данных:")
    print(f"  - Квантовых систем: {len(simulation_data['quantum_systems'])}")
    print(f"  - Точек декогеренции: {len(simulation_data['decoherence_time_series'])}")
    print(f"  - Симуляций рождения частиц: {len(simulation_data['particle_creation'])}")
    print(f"  - Параметрических разверток: {len(simulation_data['parameter_sweeps'])}")
    print(f"  - Расширенных статистик измерений: {len(simulation_data['measurement_statistics'])}")
