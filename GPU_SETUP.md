# Настройка GPU для RealitySim

RealitySim поддерживает ускорение вычислений на GPU через CuPy (для NVIDIA видеокарт).

## Установка

### 1. Проверка наличия NVIDIA GPU

```bash
# Проверка наличия CUDA
nvidia-smi
```

### 2. Установка CuPy

Выберите версию CuPy в зависимости от вашей версии CUDA:

```bash
# Для CUDA 12.x (рекомендуется)
pip install cupy-cuda12x

# Для CUDA 11.x
pip install cupy-cuda11x

# Для CUDA 10.x
pip install cupy-cuda10x

# Или установите с extras
pip install -e .[gpu]
```

### 3. Проверка установки

```python
from reality_sim.core.gpu_backend import is_gpu_available, get_device_info

print(f"GPU доступен: {is_gpu_available()}")
print(get_device_info())
```

## Использование

### Автоматическое определение

По умолчанию система автоматически использует GPU, если он доступен:

```python
from reality_sim import QuantumFabric

# Автоматически использует GPU если доступен
system = QuantumFabric(num_qubits=20)
```

### Принудительное использование CPU

```python
# Через параметр
system = QuantumFabric(num_qubits=20, use_gpu=False)

# Или через переменную окружения
import os
os.environ['REALITY_SIM_FORCE_CPU'] = '1'
```

### Принудительное использование GPU

```python
system = QuantumFabric(num_qubits=20, use_gpu=True)
```

## Преимущества GPU

- **Ускорение**: Для больших систем (20+ кубитов) GPU может ускорить вычисления в 10-100 раз
- **Больше памяти**: Современные GPU имеют 8-24 GB памяти, что позволяет работать с большими системами
- **Параллелизм**: GPU идеально подходит для параллельных операций с матрицами

## Ограничения

- **Только NVIDIA**: CuPy работает только с NVIDIA GPU с поддержкой CUDA
- **Память**: GPU память ограничена (обычно 8-24 GB)
- **Overhead**: Для малых систем (< 15 кубитов) overhead передачи данных может перевесить преимущества

## Рекомендации

- **Малые системы (2-15 кубитов)**: Используйте CPU
- **Средние системы (15-25 кубитов)**: GPU может дать значительное ускорение
- **Большие системы (25+ кубитов)**: GPU рекомендуется, но проверьте доступную память

## Примеры

```python
from reality_sim import QuantumFabric
from reality_sim.core.gpu_backend import get_device_info

# Информация об устройствах
print(get_device_info())

# Создание системы с GPU
system = QuantumFabric(num_qubits=25, use_gpu=True)
system.apply_entanglement_operator([(i, i+1) for i in range(24)])

print(f"Энтропия: {system.get_entanglement_entropy():.4f}")
```

## Устранение проблем

### CuPy не найден
```bash
pip install cupy-cuda12x  # или соответствующая версия
```

### CUDA недоступна
- Убедитесь, что установлены драйверы NVIDIA
- Проверьте версию CUDA: `nvcc --version`
- Установите соответствующую версию CuPy

### Недостаточно памяти GPU
- Уменьшите количество кубитов
- Используйте CPU для больших систем
- Освободите память GPU перед запуском

