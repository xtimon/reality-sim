# examples/quantum_emergence.py

import sys
import os
import json
import csv
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from reality_sim import QuantumFabric, EmergentLaws, Observer, HUMAN_OBSERVER, LIGO_OBSERVER, ELECTRON_OBSERVER

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
    # Для больших систем (31-50) - только полная запутанность, реже
    for num_qubits in range(35, 51, 5):  # Каждые 5 кубитов
        basic_systems_configs.append({
            'num_qubits': num_qubits,
            'entanglement_strength': 1.0
        })
    
    for config_idx, config in enumerate(basic_systems_configs):
        print(f"\n--- Система {config_idx + 1}/{len(basic_systems_configs)} ---")
        system = QuantumFabric(num_qubits=config['num_qubits'], 
                              entanglement_strength=config['entanglement_strength'])
        initial_info = {
            'num_qubits': system.n,
            'entanglement_strength': system.entanglement_strength,
            'initial_coherence': system.get_coherence(),
            'initial_entanglement': system.get_entanglement_entropy()
        }
        print(system.get_state_info())
        
        # Добавляем запутанность (все соседние пары)
        pairs = [(i, i+1) for i in range(system.n - 1)]
        system.apply_entanglement_operator(pairs)
        after_entanglement_info = {
            'coherence': system.get_coherence(),
            'entanglement_entropy': system.get_entanglement_entropy(),
            'state_info': system.get_state_info(),
            'entanglement_pairs': pairs
        }
        print(f"После запутывания ({pairs}):", system.get_state_info())
        
        # Собираем статистику измерений для всех кубитов
        print(f"\n--- Статистика измерений (10000 измерений на кубит) ---")
        all_stats = {}
        for i in range(system.n):
            stats = system.collect_measurement_statistics(i, 10000)
            all_stats[f'qubit_{i}'] = stats
            if config_idx < 3:  # Показываем детали только для первых 3 систем
                print(f"Кубит {i}: |0> = {stats['count_0']} ({stats['prob_0']:.3f}), |1> = {stats['count_1']} ({stats['prob_1']:.3f})")
        
        # Вероятности для всех кубитов
        print("\n--- Вероятности для всех кубитов ---")
        qubit_probs = {}
        for i in range(system.n):
            prob_0, prob_1 = system.get_qubit_probabilities(i)
            qubit_probs[f'qubit_{i}'] = (prob_0, prob_1)
            print(f"Кубит {i}: P(|0>) = {prob_0:.4f}, P(|1>) = {prob_1:.4f}")
        
        # Сохраняем данные
        simulation_data['quantum_systems'].append({
            **initial_info,
            **after_entanglement_info,
            'measurement_stats': all_stats,
            'qubit_probabilities': {
                k: {'prob_0': float(v[0]), 'prob_1': float(v[1])}
                for k, v in qubit_probs.items()
            }
        })
    
    return system

def demo_particle_creation():
    """Демонстрация рождения частиц из вакуума с расширенной статистикой"""
    print("\n=== Симуляция рождения пар частица-античастица ===")
    
    # Значительно расширенные множественные симуляции с разными параметрами
    vacuum_energies = np.linspace(0.01, 0.6, 30)  # 30 значений энергии
    time_steps_range = list(range(20, 501, 20))  # От 20 до 500 с шагом 20 = 25 значений
    
    all_particles_data = []
    
    for vacuum_energy in vacuum_energies:
        for time_steps in time_steps_range:
            particles = EmergentLaws.simulate_particle_creation(
                vacuum_energy=vacuum_energy, 
                time_steps=time_steps
            )
            
            # Статистика по времени создания
            creation_times = [p[0]['created_at'] for p in particles]
            
            particle_data = {
                'vacuum_energy': vacuum_energy,
                'time_steps': time_steps,
                'total_pairs': len(particles),
                'creation_times': creation_times,
                'mean_creation_time': np.mean(creation_times) if creation_times else 0,
                'std_creation_time': np.std(creation_times) if creation_times else 0,
                'first_3_pairs': [
                    {
                        'particle': p[0],
                        'antiparticle': p[1]
                    } for p in particles[:3]
                ]
            }
            all_particles_data.append(particle_data)
            
            if len(all_particles_data) % 100 == 0:  # Показываем прогресс каждые 100 симуляций
                print(f"Выполнено {len(all_particles_data)}/{len(vacuum_energies) * len(time_steps_range)} симуляций...")
    
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
    
    # Системы от 9 до 50 кубитов - используем простые паттерны
    for num_qubits in range(9, 51):
        if num_qubits <= 20:
            # Для систем до 20 кубитов - несколько паттернов
            patterns = [
                [(i, i+1) for i in range(0, num_qubits-1, 2)],  # Четные пары
                [(i, i+1) for i in range(num_qubits - 1)],  # Все соседние
            ]
            for pairs in patterns:
                configurations.append({'num_qubits': num_qubits, 'pairs': pairs, 'strength': 1.0})
        else:
            # Для больших систем (21-50) - только один простой паттерн
            pairs = [(i, i+1) for i in range(0, min(20, num_qubits-1))]  # Первые 20 пар
            configurations.append({'num_qubits': num_qubits, 'pairs': pairs, 'strength': 1.0})
    
    entanglement_data = []
    
    print(f"Выполняется анализ {len(configurations)} конфигураций...")
    for idx, config in enumerate(configurations):
        if (idx + 1) % 5 == 0:
            print(f"  Обработано {idx + 1}/{len(configurations)} конфигураций...")
            
        system = QuantumFabric(num_qubits=config['num_qubits'], 
                              entanglement_strength=config['strength'])
        initial_ent = system.get_entanglement_entropy()
        
        system.apply_entanglement_operator(config['pairs'])
        final_ent = system.get_entanglement_entropy()
        
        # Вероятности для всех кубитов
        qubit_probs = {
            i: system.get_qubit_probabilities(i) 
            for i in range(system.n)
        }
        
        data = {
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
        entanglement_data.append(data)
    
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
    sweep_data = []
    
    print(f"Выполняется sweep по {len(entanglement_strengths)} значениям силы запутанности...")
    for strength in entanglement_strengths:
        system = QuantumFabric(num_qubits=2, entanglement_strength=strength)
        system.apply_entanglement_operator([(0, 1)])
        
        sweep_data.append({
            'entanglement_strength': float(strength),
            'entanglement_entropy': float(system.get_entanglement_entropy()),
            'coherence': float(system.get_coherence()),
            'prob_0_qubit0': float(system.get_qubit_probabilities(0)[0]),
            'prob_1_qubit0': float(system.get_qubit_probabilities(0)[1]),
            'prob_0_qubit1': float(system.get_qubit_probabilities(1)[0]),
            'prob_1_qubit1': float(system.get_qubit_probabilities(1)[1])
        })
    
    print("Сила запутанности → Энтропия запутанности (выборочно):")
    for entry in sweep_data[::10]:  # Показываем каждую 10-ю точку
        print(f"  {entry['entanglement_strength']:.2f} → {entry['entanglement_entropy']:.4f}")
    
    simulation_data['parameter_sweeps'].append({
        'type': 'entanglement_strength',
        'data': sweep_data
    })
    
    # Дополнительный sweep по количеству кубитов
    print("\n=== Анализ зависимости от количества кубитов ===")
    # От 2 до 50 кубитов, с разной частотой для больших систем
    qubit_counts = list(range(2, 21))  # От 2 до 20 - все
    qubit_counts.extend(range(25, 51, 5))  # От 25 до 50 - каждые 5
    qubit_sweep_data = []
    
    for num_qubits in qubit_counts:
        # Создаем максимальную запутанность (все пары соседних кубитов)
        system = QuantumFabric(num_qubits=num_qubits, entanglement_strength=1.0)
        pairs = [(i, i+1) for i in range(num_qubits - 1)]
        system.apply_entanglement_operator(pairs)
        
        qubit_sweep_data.append({
            'num_qubits': num_qubits,
            'entanglement_pairs': len(pairs),
            'entanglement_entropy': float(system.get_entanglement_entropy()),
            'coherence': float(system.get_coherence())
        })
    
    print("Количество кубитов → Энтропия запутанности:")
    for entry in qubit_sweep_data:
        print(f"  {entry['num_qubits']} кубитов ({entry['entanglement_pairs']} пар) → {entry['entanglement_entropy']:.4f}")
    
    simulation_data['parameter_sweeps'].append({
        'type': 'qubit_count',
        'data': qubit_sweep_data
    })
    
    # Дополнительный sweep по силе запутанности для разных количеств кубитов
    print("\n=== Sweep силы запутанности для разных систем ===")
    multi_qubit_sweep = []
    # Для малых систем - детальный sweep
    for num_qubits in [2, 3, 4, 5]:
        for strength in np.linspace(0.1, 1.0, 50):
            system = QuantumFabric(num_qubits=num_qubits, entanglement_strength=strength)
            pairs = [(i, i+1) for i in range(num_qubits - 1)]
            system.apply_entanglement_operator(pairs)
            
            multi_qubit_sweep.append({
                'num_qubits': num_qubits,
                'entanglement_strength': float(strength),
                'entanglement_entropy': float(system.get_entanglement_entropy()),
                'coherence': float(system.get_coherence())
            })
    # Для больших систем - только несколько значений силы
    for num_qubits in [10, 20, 30, 40, 50]:
        for strength in [0.25, 0.5, 0.75, 1.0]:
            system = QuantumFabric(num_qubits=num_qubits, entanglement_strength=strength)
            pairs = [(i, i+1) for i in range(min(20, num_qubits - 1))]  # Ограничиваем количество пар для больших систем
            system.apply_entanglement_operator(pairs)
            
            multi_qubit_sweep.append({
                'num_qubits': num_qubits,
                'entanglement_strength': float(strength),
                'entanglement_entropy': float(system.get_entanglement_entropy()),
                'coherence': float(system.get_coherence())
            })
    
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
    
    # Большие системы (31-50) - минимальные конфигурации
    for num_qubits in range(35, 51, 5):  # 35, 40, 45, 50
        pairs = [(i, i+1) for i in range(min(20, num_qubits - 1))]  # Ограничиваем количество пар
        measurement_configs.append({
            'num_qubits': num_qubits,
            'num_measurements': 10000,  # Меньше измерений для больших систем
            'pairs': pairs
        })
    
    measurement_data = []
    
    print(f"Выполняется {len(measurement_configs)} конфигураций измерений...")
    for idx, config in enumerate(measurement_configs):
        if (idx + 1) % 10 == 0:
            print(f"  Обработано {idx + 1}/{len(measurement_configs)} конфигураций...")
            
        system = QuantumFabric(num_qubits=config['num_qubits'])
        system.apply_entanglement_operator(config['pairs'])
        
        stats_per_qubit = {}
        for i in range(system.n):
            stats = system.collect_measurement_statistics(i, config['num_measurements'])
            stats_per_qubit[f'qubit_{i}'] = stats
        
        measurement_data.append({
            'num_qubits': config['num_qubits'],
            'entanglement_pairs': config['pairs'],
            'num_measurements': config['num_measurements'],
            'measurement_stats': stats_per_qubit,
            'coherence': float(system.get_coherence()),
            'entanglement_entropy': float(system.get_entanglement_entropy())
        })
        
        if idx < 5:  # Показываем детали только для первых 5
            print(f"\n{config['num_qubits']} кубитов, {config['num_measurements']} измерений:")
            for i in range(system.n):
                stats = stats_per_qubit[f'qubit_{i}']
                print(f"  Кубит {i}: P(|0>)={stats['prob_0']:.4f}, P(|1>)={stats['prob_1']:.4f}")
    
    simulation_data['measurement_statistics'].extend(measurement_data)

if __name__ == "__main__":
    print("=" * 60)
    print("РАСШИРЕННАЯ СИМУЛЯЦИЯ КВАНТОВОЙ ЭМЕРДЖЕНТНОСТИ")
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
    # Анализ систем от 10 до 50 кубитов
    large_qubit_counts = list(range(10, 21)) + list(range(25, 51, 5))  # 10-20 все, 25-50 каждые 5
    
    print(f"Анализ {len(large_qubit_counts)} больших систем...")
    for idx, num_qubits in enumerate(large_qubit_counts):
        if (idx + 1) % 5 == 0:
            print(f"  Обработано {idx + 1}/{len(large_qubit_counts)} систем...")
            
        system = QuantumFabric(num_qubits=num_qubits, entanglement_strength=1.0)
        # Для больших систем ограничиваем количество пар запутанности
        max_pairs = min(30, num_qubits - 1)
        pairs = [(i, i+1) for i in range(max_pairs)]
        system.apply_entanglement_operator(pairs)
        
        large_systems.append({
            'num_qubits': num_qubits,
            'entanglement_pairs': len(pairs),
            'entanglement_entropy': float(system.get_entanglement_entropy()),
            'coherence': float(system.get_coherence())
        })
        
        if idx < 5 or (idx + 1) % 5 == 0:  # Показываем первые 5 и каждую 5-ю
            print(f"{num_qubits} кубитов ({len(pairs)} пар): энтропия = {large_systems[-1]['entanglement_entropy']:.4f}")
    
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
