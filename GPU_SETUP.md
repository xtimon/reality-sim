# Настройка GPU для RealitySim

RealitySim поддерживает ускорение вычислений на GPU через:
- **CuPy** (для NVIDIA видеокарт с CUDA)
- **vulkpy** (для AMD, NVIDIA и Intel видеокарт с Vulkan)
- **PyOpenCL** (для AMD, NVIDIA и Intel видеокарт с OpenCL)

Система автоматически выберет лучший доступный backend: CUDA (приоритет) > Vulkan > OpenCL > CPU.

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

### Вариант 2: AMD/NVIDIA/Intel GPU (Vulkan)

#### 1. Проверка наличия Vulkan

```bash
# Для Linux
vulkaninfo  # Может потребоваться установка: sudo pacman -S vulkan-tools (Arch) или sudo apt-get install vulkan-tools (Ubuntu/Debian)

# Для macOS
# Vulkan поддерживается через MoltenVK
```

#### 2. Установка vulkpy

```bash
# Установка vulkpy
pip install vulkpy

# Или установите с extras
pip install -e .[vulkan]

# Или установите все GPU зависимости (CUDA + Vulkan + OpenCL)
pip install -e .[gpu-all]
```

#### 3. Драйверы для Vulkan

**Linux:**
- **Mesa (открытый драйвер)** - рекомендуется для большинства случаев
  ```bash
  # Ubuntu/Debian
  sudo apt-get install mesa-vulkan-drivers vulkan-tools
  
  # Fedora
  sudo dnf install mesa-vulkan-drivers vulkan-tools
  
  # Arch Linux
  sudo pacman -S vulkan-radeon vulkan-tools  # Для AMD
  sudo pacman -S vulkan-intel vulkan-tools   # Для Intel
  sudo pacman -S nvidia-utils vulkan-tools  # Для NVIDIA
  ```

**macOS:**
- Vulkan поддерживается через MoltenVK (обычно устанавливается с vulkpy)

**Примечание:** Vulkan может быть быстрее OpenCL на некоторых системах, особенно на AMD GPU с Mesa драйверами.

### Вариант 3: AMD/NVIDIA/Intel GPU (OpenCL)

#### 1. Проверка наличия OpenCL

```bash
# Для Linux (AMD)
clinfo  # Может потребоваться установка: sudo pacman -S clinfo (Arch) или sudo apt-get install clinfo (Ubuntu/Debian)

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

#### 3. Драйверы для AMD GPU

**Linux (AMD):**

RealitySim поддерживает **оба** варианта драйверов для AMD GPU:

1. **Mesa (открытый драйвер)** - рекомендуется для большинства случаев
   ```bash
   # Ubuntu/Debian
   sudo apt-get install mesa-opencl-icd opencl-headers ocl-icd-opencl-dev
   
   # Fedora
   sudo dnf install mesa-libOpenCL opencl-headers ocl-icd
   
   # Arch Linux
   sudo pacman -S opencl-mesa opencl-headers ocl-icd
   ```
   Mesa предоставляет OpenCL через `Rusticl` (новые версии) или `Clover` (старые версии).

2. **AMDGPU-PRO (проприетарный)** - для максимальной производительности
   - Скачайте с официального сайта AMD
   - Обычно обеспечивает лучшую производительность, но может быть сложнее в установке

**macOS:** OpenCL обычно уже установлен в системе.

**Примечание:** Система автоматически определит доступный OpenCL драйвер (Mesa или проприетарный) и будет использовать его.

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
- Для Vulkan: `vulkan_available: True`, `vulkan_device_name`, `vulkan_device_type`
- Для AMD/OpenCL: `opencl_available: True`, `opencl_platforms` с информацией о платформах и устройствах
- Для Mesa: в `opencl_platforms` будет указано `is_mesa: True` и название платформы будет содержать "mesa", "clover" или "rusticl"

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
- **Больше памяти**: Современные GPU имеют 4-24 GB памяти, что позволяет работать с большими системами
- **Параллелизм**: GPU идеально подходит для параллельных операций с матрицами
- **Кроссплатформенность**: OpenCL работает на различных GPU (NVIDIA, AMD, Intel) и операционных системах

## Ограничения

- **Память**: GPU память ограничена (обычно 4-24 GB)
- **Overhead**: Для малых систем (< 15 кубитов) overhead передачи данных может перевесить преимущества
- **OpenCL производительность**: 
  - OpenCL может быть медленнее CUDA на NVIDIA GPU, но является единственным вариантом для AMD GPU
  - Mesa драйвер может быть медленнее проприетарных драйверов AMD, но более стабилен и проще в установке
  - Для максимальной производительности на AMD рекомендуется AMDGPU-PRO, но Mesa также полностью поддерживается

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

### AMD/NVIDIA/Intel GPU (Vulkan)

#### vulkpy не найден
```bash
pip install vulkpy
```

#### Vulkan недоступен

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install mesa-vulkan-drivers vulkan-tools

# Fedora
sudo dnf install mesa-vulkan-drivers vulkan-tools

# Arch Linux
sudo pacman -S vulkan-radeon vulkan-tools  # Для AMD
sudo pacman -S vulkan-intel vulkan-tools   # Для Intel
sudo pacman -S nvidia-utils vulkan-tools   # Для NVIDIA
```

**macOS:**
- MoltenVK обычно устанавливается автоматически с vulkpy

**Проверка установки:**
```bash
# Через vulkaninfo (Linux)
vulkaninfo

# Или через Python
python3 -c "
from reality_sim.core.gpu_backend import get_device_info
import json
print(json.dumps(get_device_info(), indent=2, default=str))
"
```

### AMD GPU (OpenCL)

#### PyOpenCL не найден
```bash
pip install pyopencl
```

#### OpenCL недоступен

**Linux (AMD) с Mesa:**

```bash
# Ubuntu/Debian
sudo apt-get install mesa-opencl-icd opencl-headers ocl-icd-opencl-dev

# Fedora
sudo dnf install mesa-libOpenCL opencl-headers ocl-icd

# Arch Linux
sudo pacman -S opencl-mesa opencl-headers ocl-icd

# Проверка установки
clinfo
```

**Linux (AMD) с AMDGPU-PRO:**
- Скачайте и установите драйверы с официального сайта AMD
- Убедитесь, что OpenCL runtime установлен

**macOS:** OpenCL обычно уже установлен

**Проверка доступных устройств:**
```bash
# Через clinfo (Linux)
clinfo

# Или через Python
python3 -c "
from reality_sim.core.gpu_backend import get_device_info
import json
print(json.dumps(get_device_info(), indent=2, default=str))
"
```

**Определение типа драйвера:**
```python
import pyopencl as cl
platforms = cl.get_platforms()
for platform in platforms:
    print(f"Platform: {platform.name}")
    print(f"  Vendor: {platform.vendor}")
    # Mesa обычно содержит 'mesa', 'clover' или 'rusticl' в названии
    for device in platform.get_devices():
        print(f"  Device: {device.name}")
        print(f"    Type: {device.type}")
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

