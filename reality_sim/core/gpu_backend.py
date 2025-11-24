# reality_sim/core/gpu_backend.py

"""
Модуль для автоматического выбора CPU/GPU backend для вычислений.
Поддерживает CuPy (NVIDIA GPU) и NumPy (CPU).
"""

import os
import warnings
from typing import Optional

# Попытка импортировать CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False
    CUDA_AVAILABLE = False
    cp = None

# Импорт NumPy как fallback
import numpy as np

# Глобальная переменная для принудительного использования CPU
_FORCE_CPU = os.getenv('REALITY_SIM_FORCE_CPU', '0').lower() in ('1', 'true', 'yes')

class Backend:
    """
    Класс для управления backend вычислений (CPU/GPU).
    """
    
    def __init__(self, use_gpu: Optional[bool] = None):
        """
        Инициализирует backend.
        
        Args:
            use_gpu: Если True - использовать GPU, False - CPU, None - автоопределение
        """
        if use_gpu is None:
            # Автоопределение: используем GPU если доступен и не принудительно CPU
            self.use_gpu = CUDA_AVAILABLE and not _FORCE_CPU
        else:
            self.use_gpu = use_gpu and CUDA_AVAILABLE
        
        if self.use_gpu and not CUDA_AVAILABLE:
            warnings.warn("GPU запрошен, но CuPy/CUDA недоступны. Используется CPU.")
            self.use_gpu = False
        
        self.xp = cp if self.use_gpu else np
        self._device = 'GPU' if self.use_gpu else 'CPU'
    
    def __str__(self):
        return f"Backend({self._device})"
    
    def get_array_module(self):
        """Возвращает модуль для работы с массивами (cupy или numpy)"""
        return self.xp
    
    def to_cpu(self, array):
        """Конвертирует массив в CPU (numpy)"""
        if self.use_gpu and hasattr(array, 'get'):
            return array.get()  # CuPy array -> NumPy
        return array
    
    def to_gpu(self, array):
        """Конвертирует массив в GPU (cupy)"""
        if self.use_gpu:
            return cp.asarray(array)
        return array
    
    def asarray(self, obj, dtype=None):
        """Создает массив используя текущий backend"""
        return self.xp.asarray(obj, dtype=dtype)
    
    def zeros(self, shape, dtype=None):
        """Создает массив нулей"""
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        """Создает массив единиц"""
        return self.xp.ones(shape, dtype=dtype)
    
    def array(self, obj, dtype=None):
        """Создает массив из объекта"""
        return self.xp.array(obj, dtype=dtype)
    
    def copy(self, array):
        """Копирует массив"""
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
    """Проверяет доступность GPU"""
    return CUDA_AVAILABLE

def get_device_info() -> dict:
    """Возвращает информацию о доступных устройствах"""
    info = {
        'cupy_available': CUPY_AVAILABLE,
        'cuda_available': CUDA_AVAILABLE,
        'force_cpu': _FORCE_CPU,
        'current_backend': str(_default_backend)
    }
    
    if CUDA_AVAILABLE:
        try:
            info['gpu_count'] = cp.cuda.runtime.getDeviceCount()
            info['gpu_name'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            info['gpu_memory'] = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
        except:
            pass
    
    return info

