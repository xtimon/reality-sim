# Настройка GPU для RealitySim

RealitySim поддерживает ускорение вычислений на GPU через:
- **CuPy** (для NVIDIA видеокарт с CUDA)
- **PyOpenCL** (для AMD, NVIDIA и Intel видеокарт с OpenCL)

Система автоматически выберет лучший доступный backend: CUDA (приоритет) > OpenCL > CPU.

## Установка

### Вариант 1: NVIDIA GPU (CUDA)

#### 1. Проверка наличия NVIDIA GPU

```bash
# Проверка наличия CUDA
nvidia-smi
```

#### 2. Установка CuPy

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

### Вариант 2: AMD/NVIDIA/Intel GPU (OpenCL)

#### 1. Проверка наличия OpenCL

```bash
# Для Linux (AMD)
clinfo

# Для macOS
system_profiler SPDisplaysDataType
```

#### 2. Установка PyOpenCL

```bash
# Установка PyOpenCL
pip install pyopencl

# Или установите с extras
pip install -e .[opencl]

# Или установите все GPU зависимости (CUDA + OpenCL)
pip install -e .[gpu-all]
```

**Примечание для macOS:** OpenCL обычно уже установлен в системе.

**Примечание для Linux (AMD):** Убедитесь, что установлены драйверы AMD с поддержкой OpenCL.

### 3. Проверка установки

```python
from reality_sim.core.gpu_backend import is_gpu_available, get_device_info

print(f"GPU доступен: {is_gpu_available()}")
info = get_device_info()
print("Информация об устройствах:")
for k, v in info.items():
    print(f"  {k}: {v}")
```

Вывод покажет:
- Для NVIDIA: `cuda_available: True`, `cuda_gpu_name`, `cuda_gpu_memory`
- Для AMD/OpenCL: `opencl_available: True`, `opencl_platforms` с информацией о платформах и устройствах

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

- **Память**: GPU память ограничена (обычно 4-24 GB)
- **Overhead**: Для малых систем (< 15 кубитов) overhead передачи данных может перевесить преимущества
- **OpenCL производительность**: OpenCL может быть медленнее CUDA на NVIDIA GPU, но является единственным вариантом для AMD GPU

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

### NVIDIA GPU

#### CuPy не найден
```bash
pip install cupy-cuda12x  # или соответствующая версия
```

#### CUDA недоступна
- Убедитесь, что установлены драйверы NVIDIA
- Проверьте версию CUDA: `nvcc --version`
- Установите соответствующую версию CuPy

### AMD GPU (OpenCL)

#### PyOpenCL не найден
```bash
pip install pyopencl
```

#### OpenCL недоступен
- **Linux (AMD)**: Установите драйверы AMD с поддержкой OpenCL
  ```bash
  # Ubuntu/Debian
  sudo apt-get install opencl-headers ocl-icd-opencl-dev
  ```
- **macOS**: OpenCL обычно уже установлен
- Проверьте доступные устройства: `clinfo` (Linux) или через Python:
  ```python
  import pyopencl as cl
  platforms = cl.get_platforms()
  for platform in platforms:
      print(f"Platform: {platform.name}")
      for device in platform.get_devices():
          print(f"  Device: {device.name}")
  ```

### Общие проблемы

#### Недостаточно памяти GPU
- Уменьшите количество кубитов
- Используйте CPU для больших систем
- Освободите память GPU перед запуском

#### GPU не используется автоматически
- Проверьте доступность: `is_gpu_available()`
- Принудительно используйте CPU: `use_gpu=False`
- Проверьте переменную окружения: `REALITY_SIM_FORCE_CPU`

