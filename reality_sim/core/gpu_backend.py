# reality_sim/core/gpu_backend.py

"""
Модуль для автоматического выбора CPU/GPU backend для вычислений.
Поддерживает:
- CuPy (NVIDIA GPU через CUDA) - приоритет 1
- Vulkan (AMD/NVIDIA/Intel GPU через Vulkan) - приоритет 2
- PyOpenCL (AMD/NVIDIA/Intel GPU через OpenCL) - приоритет 3
- NumPy (CPU) - fallback
"""

import os
import warnings
from typing import Optional, Union

# Попытка импортировать CuPy (NVIDIA CUDA)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False
    CUDA_AVAILABLE = False
    cp = None

# Попытка импортировать Vulkan (AMD/NVIDIA/Intel через Vulkan)
try:
    import vulkpy as vk
    VULKPY_AVAILABLE = True
    VULKAN_AVAILABLE = True  # Будет проверено при инициализации
    _vulkan_context = None
except ImportError:
    VULKPY_AVAILABLE = False
    VULKAN_AVAILABLE = False
    vk = None
    _vulkan_context = None
except Exception as e:
    VULKPY_AVAILABLE = False
    VULKAN_AVAILABLE = False
    vk = None
    _vulkan_context = None

# Попытка импортировать PyOpenCL (AMD/NVIDIA/Intel OpenCL)
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    PYOPENCL_AVAILABLE = True
    OPENCL_AVAILABLE = len(cl.get_platforms()) > 0
    _opencl_context = None
    _opencl_queue = None
except ImportError:
    PYOPENCL_AVAILABLE = False
    OPENCL_AVAILABLE = False
    cl = None
    cl_array = None
    _opencl_context = None
    _opencl_queue = None
except Exception as e:
    PYOPENCL_AVAILABLE = False
    OPENCL_AVAILABLE = False
    cl = None
    cl_array = None
    _opencl_context = None
    _opencl_queue = None

# Импорт NumPy как fallback
import numpy as np

# Глобальная переменная для принудительного использования CPU
_FORCE_CPU = os.getenv('REALITY_SIM_FORCE_CPU', '0').lower() in ('1', 'true', 'yes')

# Приоритет GPU: CUDA > Vulkan > OpenCL > CPU
GPU_AVAILABLE = CUDA_AVAILABLE or VULKAN_AVAILABLE or OPENCL_AVAILABLE

class VulkanBackend:
    """
    Обертка для Vulkan через vulkpy, предоставляющая NumPy-подобный интерфейс.
    Использует гибридный подход: простые операции на GPU, сложные - на CPU с конвертацией.
    """
    def __init__(self, context):
        self.context = context
    
    def _to_cpu(self, x):
        """Конвертирует в CPU массив"""
        if isinstance(x, vk.Array):
            return x.numpy()
        return x
    
    def _to_gpu(self, x):
        """Конвертирует в GPU массив"""
        if isinstance(x, vk.Array):
            return x
        return vk.Array(self.context, np.asarray(x))
    
    def zeros(self, shape, dtype=None):
        """Создает массив нулей на Vulkan устройстве"""
        if dtype is None:
            dtype = np.complex128
        return vk.zeros(self.context, shape, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        """Создает массив единиц на Vulkan устройстве"""
        if dtype is None:
            dtype = np.complex128
        return vk.ones(self.context, shape, dtype=dtype)
    
    def array(self, obj, dtype=None):
        """Создает массив из объекта"""
        return vk.Array(self.context, np.asarray(obj, dtype=dtype))
    
    def asarray(self, obj, dtype=None):
        """Создает массив из объекта"""
        if isinstance(obj, vk.Array):
            return obj
        return vk.Array(self.context, np.asarray(obj, dtype=dtype))
    
    def copy(self, array):
        """Копирует массив"""
        if isinstance(array, vk.Array):
            return array.copy()
        return np.copy(array)
    
    # NumPy-подобные функции через numpy с конвертацией
    @property
    def linalg(self):
        """Линейная алгебра - используем numpy с конвертацией"""
        class VulkanLinalg:
            def __init__(self, backend):
                self.backend = backend
            
            def norm(self, x):
                if isinstance(x, vk.Array):
                    return float(np.linalg.norm(x.numpy()))
                return float(np.linalg.norm(x))
            
            def svd(self, A):
                A_cpu = A.numpy() if isinstance(A, vk.Array) else A
                U, s, Vh = np.linalg.svd(A_cpu)
                return (
                    vk.Array(self.backend.context, U),
                    s,
                    vk.Array(self.backend.context, Vh)
                )
            
            def eigvals(self, A):
                A_cpu = A.numpy() if isinstance(A, vk.Array) else A
                return np.linalg.eigvals(A_cpu)
        
        return VulkanLinalg(self)
    
    @property
    def pi(self):
        return np.pi
    
    def cos(self, x):
        """Косинус - используем Vulkan операции если возможно"""
        if isinstance(x, vk.Array):
            # Vulkan операции через конвертацию
            return vk.Array(self.context, np.cos(x.numpy()))
        return np.cos(x)
    
    def sin(self, x):
        """Синус - используем Vulkan операции если возможно"""
        if isinstance(x, vk.Array):
            return vk.Array(self.context, np.sin(x.numpy()))
        return np.sin(x)
    
    def sqrt(self, x):
        """Квадратный корень"""
        if isinstance(x, (int, float)):
            return np.sqrt(x)
        if isinstance(x, vk.Array):
            return vk.Array(self.context, np.sqrt(x.numpy()))
        return np.sqrt(x)
    
    def abs(self, x):
        """Модуль"""
        if isinstance(x, vk.Array):
            return vk.Array(self.context, np.abs(x.numpy()))
        return np.abs(x)
    
    def sum(self, x):
        """Сумма элементов"""
        if isinstance(x, vk.Array):
            return float(np.sum(x.numpy()))
        return np.sum(x)
    
    def mean(self, x):
        """Среднее значение"""
        if isinstance(x, vk.Array):
            return float(np.mean(x.numpy()))
        return np.mean(x)
    
    def real(self, x):
        """Действительная часть"""
        if isinstance(x, vk.Array):
            return vk.Array(self.context, np.real(x.numpy()))
        return np.real(x)
    
    def log2(self, x):
        """Логарифм по основанию 2"""
        if isinstance(x, vk.Array):
            return np.log2(x.numpy())
        return np.log2(x)
    
    def eye(self, n, dtype=None):
        """Создает единичную матрицу"""
        if dtype is None:
            dtype = np.complex128
        I = np.eye(n, dtype=dtype)
        return vk.Array(self.context, I)
    
    def allclose(self, a, b, atol=1e-10):
        """Проверяет близость массивов"""
        a_cpu = a.numpy() if isinstance(a, vk.Array) else a
        b_cpu = b.numpy() if isinstance(b, vk.Array) else b
        return np.allclose(a_cpu, b_cpu, atol=atol)
    
    def outer(self, a, b):
        """Внешнее произведение"""
        a_cpu = a.numpy() if isinstance(a, vk.Array) else a
        b_cpu = b.numpy() if isinstance(b, vk.Array) else b
        result = np.outer(a_cpu, b_cpu)
        return vk.Array(self.context, result)

class OpenCLBackend:
    """
    Обертка для PyOpenCL, предоставляющая NumPy-подобный интерфейс.
    Использует гибридный подход: простые операции на GPU, сложные - на CPU с конвертацией.
    """
    def __init__(self, context, queue):
        self.context = context
        self.queue = queue
    
    def _to_cpu(self, x):
        """Конвертирует в CPU массив"""
        if hasattr(x, 'get'):
            return x.get()
        return x
    
    def _to_gpu(self, x):
        """Конвертирует в GPU массив"""
        if isinstance(x, cl_array.Array):
            return x
        return cl_array.to_device(self.queue, np.asarray(x))
    
    def zeros(self, shape, dtype=None):
        """Создает массив нулей на OpenCL устройстве"""
        if dtype is None:
            dtype = np.complex128
        return cl_array.zeros(self.queue, shape, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        """Создает массив единиц на OpenCL устройстве"""
        if dtype is None:
            dtype = np.complex128
        return cl_array.ones(self.queue, shape, dtype=dtype)
    
    def array(self, obj, dtype=None):
        """Создает массив из объекта"""
        return cl_array.to_device(self.queue, np.asarray(obj, dtype=dtype))
    
    def asarray(self, obj, dtype=None):
        """Создает массив из объекта"""
        if isinstance(obj, cl_array.Array):
            return obj
        return cl_array.to_device(self.queue, np.asarray(obj, dtype=dtype))
    
    def copy(self, array):
        """Копирует массив"""
        if isinstance(array, cl_array.Array):
            return array.copy()
        return np.copy(array)
    
    # NumPy-подобные функции через numpy с конвертацией
    @property
    def linalg(self):
        """Линейная алгебра - используем numpy с конвертацией"""
        class OpenCLLinalg:
            def __init__(self, backend):
                self.backend = backend
            
            def norm(self, x):
                return float(np.linalg.norm(x.get()))
            
            def svd(self, A):
                A_cpu = A.get()
                U, s, Vh = np.linalg.svd(A_cpu)
                return (
                    cl_array.to_device(self.backend.queue, U),
                    s,
                    cl_array.to_device(self.backend.queue, Vh)
                )
            
            def eigvals(self, A):
                A_cpu = A.get()
                return np.linalg.eigvals(A_cpu)
        
        return OpenCLLinalg(self)
    
    @property
    def pi(self):
        return np.pi
    
    def cos(self, x):
        """Косинус - используем OpenCL операции если возможно"""
        if isinstance(x, cl_array.Array):
            # Используем OpenCL операции
            return cl_array.cos(x)
        return np.cos(x)
    
    def sin(self, x):
        """Синус - используем OpenCL операции если возможно"""
        if isinstance(x, cl_array.Array):
            return cl_array.sin(x)
        return np.sin(x)
    
    def sqrt(self, x):
        """Квадратный корень"""
        if isinstance(x, (int, float)):
            return np.sqrt(x)
        if isinstance(x, cl_array.Array):
            return cl_array.sqrt(x)
        return np.sqrt(x)
    
    def abs(self, x):
        """Модуль"""
        if isinstance(x, cl_array.Array):
            return cl_array.abs(x)
        return np.abs(x)
    
    def sum(self, x):
        """Сумма элементов"""
        if isinstance(x, cl_array.Array):
            return float(cl_array.sum(x).get())
        return np.sum(x)
    
    def mean(self, x):
        """Среднее значение"""
        if isinstance(x, cl_array.Array):
            return float(cl_array.mean(x).get())
        return np.mean(x)
    
    def real(self, x):
        """Действительная часть"""
        if isinstance(x, cl_array.Array):
            return cl_array.real(x)
        return np.real(x)
    
    def log2(self, x):
        """Логарифм по основанию 2"""
        if isinstance(x, cl_array.Array):
            # OpenCL не имеет log2, используем CPU
            return np.log2(x.get())
        return np.log2(x)
    
    def eye(self, n, dtype=None):
        """Создает единичную матрицу"""
        if dtype is None:
            dtype = np.complex128
        I = np.eye(n, dtype=dtype)
        return cl_array.to_device(self.queue, I)
    
    def allclose(self, a, b, atol=1e-10):
        """Проверяет близость массивов"""
        a_cpu = a.get() if hasattr(a, 'get') else a
        b_cpu = b.get() if hasattr(b, 'get') else b
        return np.allclose(a_cpu, b_cpu, atol=atol)
    
    def outer(self, a, b):
        """Внешнее произведение"""
        a_cpu = a.get() if hasattr(a, 'get') else a
        b_cpu = b.get() if hasattr(b, 'get') else b
        result = np.outer(a_cpu, b_cpu)
        return cl_array.to_device(self.queue, result)

class Backend:
    """
    Класс для управления backend вычислений (CPU/GPU).
    Поддерживает CUDA (NVIDIA), OpenCL (AMD/NVIDIA/Intel) и CPU.
    """
    
    def __init__(self, use_gpu: Optional[bool] = None, prefer_cuda: bool = True):
        """
        Инициализирует backend.
        
        Args:
            use_gpu: Если True - использовать GPU, False - CPU, None - автоопределение
            prefer_cuda: Если True - предпочитать CUDA над OpenCL (для NVIDIA)
        """
        self.use_cuda = False
        self.use_opencl = False
        self.opencl_backend = None
        
        if use_gpu is None:
            # Автоопределение: используем GPU если доступен и не принудительно CPU
            use_gpu = GPU_AVAILABLE and not _FORCE_CPU
        
        if use_gpu:
            # Приоритет: CUDA > OpenCL
            if prefer_cuda and CUDA_AVAILABLE:
                self.use_cuda = True
                self.xp = cp
                self._device = 'CUDA (NVIDIA)'
            elif OPENCL_AVAILABLE:
                self.use_opencl = True
                self._init_opencl()
                self.xp = self.opencl_backend
                self._device = 'OpenCL (AMD/NVIDIA/Intel)'
            else:
                if use_gpu:
                    warnings.warn("GPU запрошен, но ни CUDA, ни OpenCL недоступны. Используется CPU.")
                self.xp = np
                self._device = 'CPU'
        else:
            self.xp = np
            self._device = 'CPU'
    
    def _init_opencl(self):
        """Инициализирует OpenCL контекст"""
        global _opencl_context, _opencl_queue
        
        if _opencl_context is None:
            try:
                platforms = cl.get_platforms()
                gpu_devices = []
                cpu_devices = []
                
                # Собираем все доступные устройства, разделяя на GPU и CPU
                for platform in platforms:
                    for device in platform.get_devices(cl.device_type.GPU):
                        gpu_devices.append((platform, device))
                    for device in platform.get_devices(cl.device_type.CPU):
                        cpu_devices.append((platform, device))
                
                # Предпочитаем GPU устройства (включая Mesa)
                if gpu_devices:
                    # Выбираем первое GPU устройство
                    # Mesa обычно предоставляет GPU через платформу "Clover" или "Rusticl"
                    platform, device = gpu_devices[0]
                    _opencl_context = cl.Context(devices=[device])
                    _opencl_queue = cl.CommandQueue(_opencl_context)
                elif cpu_devices:
                    # Fallback на CPU если GPU нет
                    warnings.warn("GPU устройства не найдены, используется CPU OpenCL")
                    platform, device = cpu_devices[0]
                    _opencl_context = cl.Context(devices=[device])
                    _opencl_queue = cl.CommandQueue(_opencl_context)
                else:
                    raise RuntimeError("Нет доступных OpenCL устройств")
            except Exception as e:
                warnings.warn(f"Не удалось инициализировать OpenCL: {e}")
                raise
        
        self.opencl_backend = OpenCLBackend(_opencl_context, _opencl_queue)
    
    def __str__(self):
        return f"Backend({self._device})"
    
    def get_array_module(self):
        """Возвращает модуль для работы с массивами (cupy, vulkan, opencl или numpy)"""
        return self.xp
    
    def to_cpu(self, array):
        """Конвертирует массив в CPU (numpy)"""
        if self.use_cuda and hasattr(array, 'get'):
            return array.get()  # CuPy array -> NumPy
        elif self.use_vulkan and isinstance(array, vk.Array):
            return array.numpy()  # Vulkan array -> NumPy
        elif self.use_opencl and hasattr(array, 'get'):
            return array.get()  # OpenCL array -> NumPy
        return array
    
    def to_gpu(self, array):
        """Конвертирует массив в GPU"""
        if self.use_cuda:
            return cp.asarray(array)
        elif self.use_vulkan:
            return self.vulkan_backend.asarray(array)
        elif self.use_opencl:
            return self.opencl_backend.asarray(array)
        return array
    
    def asarray(self, obj, dtype=None):
        """Создает массив используя текущий backend"""
        if self.use_vulkan:
            return self.vulkan_backend.asarray(obj, dtype=dtype)
        elif self.use_opencl:
            return self.opencl_backend.asarray(obj, dtype=dtype)
        return self.xp.asarray(obj, dtype=dtype)
    
    def zeros(self, shape, dtype=None):
        """Создает массив нулей"""
        if self.use_vulkan:
            return self.vulkan_backend.zeros(shape, dtype=dtype)
        elif self.use_opencl:
            return self.opencl_backend.zeros(shape, dtype=dtype)
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        """Создает массив единиц"""
        if self.use_vulkan:
            return self.vulkan_backend.ones(shape, dtype=dtype)
        elif self.use_opencl:
            return self.opencl_backend.ones(shape, dtype=dtype)
        return self.xp.ones(shape, dtype=dtype)
    
    def array(self, obj, dtype=None):
        """Создает массив из объекта"""
        if self.use_vulkan:
            return self.vulkan_backend.array(obj, dtype=dtype)
        elif self.use_opencl:
            return self.opencl_backend.array(obj, dtype=dtype)
        return self.xp.array(obj, dtype=dtype)
    
    def copy(self, array):
        """Копирует массив"""
        if self.use_vulkan:
            return self.vulkan_backend.copy(array)
        elif self.use_opencl:
            return self.opencl_backend.copy(array)
        return self.xp.copy(array)

# Глобальный backend (автоопределение)
_default_backend = Backend()

def get_backend(use_gpu: Optional[bool] = None) -> Backend:
    """
    Получить backend для вычислений.
    
    Args:
        use_gpu: Если True - использовать GPU, False - CPU, None - автоопределение
    
    Returns:
        Backend объект
    """
    if use_gpu is None:
        return _default_backend
    return Backend(use_gpu=use_gpu)

def set_default_backend(use_gpu: Optional[bool] = None):
    """
    Установить backend по умолчанию.
    
    Args:
        use_gpu: Если True - использовать GPU, False - CPU, None - автоопределение
    """
    global _default_backend
    _default_backend = Backend(use_gpu=use_gpu)

def is_gpu_available() -> bool:
    """Проверяет доступность GPU (CUDA или OpenCL)"""
    return GPU_AVAILABLE

def get_device_info() -> dict:
    """Возвращает информацию о доступных устройствах"""
    info = {
        'cupy_available': CUPY_AVAILABLE,
        'cuda_available': CUDA_AVAILABLE,
        'vulkpy_available': VULKPY_AVAILABLE,
        'vulkan_available': VULKAN_AVAILABLE,
        'pyopencl_available': PYOPENCL_AVAILABLE,
        'opencl_available': OPENCL_AVAILABLE,
        'gpu_available': GPU_AVAILABLE,
        'force_cpu': _FORCE_CPU,
        'current_backend': str(_default_backend)
    }
    
    # Информация о CUDA
    if CUDA_AVAILABLE:
        try:
            info['cuda_gpu_count'] = cp.cuda.runtime.getDeviceCount()
            info['cuda_gpu_name'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            info['cuda_gpu_memory'] = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
        except:
            pass
    
    # Информация о Vulkan
    if VULKAN_AVAILABLE:
        try:
            context = vk.Context()
            if hasattr(context, 'device'):
                device = context.device
                info['vulkan_device_name'] = getattr(device, 'name', 'Unknown')
                info['vulkan_device_type'] = str(getattr(device, 'type', 'Unknown'))
                info['vulkan_driver_version'] = getattr(device, 'driver_version', None)
        except:
            pass
    
    # Информация об OpenCL
    if OPENCL_AVAILABLE:
        try:
            platforms = cl.get_platforms()
            info['opencl_platforms'] = []
            for i, platform in enumerate(platforms):
                platform_info = {
                    'name': platform.name,
                    'vendor': platform.vendor,
                    'version': platform.version if hasattr(platform, 'version') else None,
                    'devices': []
                }
                for device in platform.get_devices():
                    device_info = {
                        'name': device.name,
                        'type': str(device.type),
                        'vendor': device.vendor if hasattr(device, 'vendor') else None,
                        'driver_version': device.driver_version if hasattr(device, 'driver_version') else None,
                        'memory': device.global_mem_size if hasattr(device, 'global_mem_size') else None,
                        'is_mesa': 'mesa' in platform.name.lower() or 'clover' in platform.name.lower() or 'rusticl' in platform.name.lower()
                    }
                    platform_info['devices'].append(device_info)
                info['opencl_platforms'].append(platform_info)
        except:
            pass
    
    return info

