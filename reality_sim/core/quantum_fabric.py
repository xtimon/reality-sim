# reality_sim/core/quantum_fabric.py

import numpy as np
from typing import List, Tuple, Optional
import warnings

# Импорт GPU backend (опционально)
try:
    from .gpu_backend import get_backend, Backend
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    def get_backend(use_gpu=None):
        class DummyBackend:
            def get_array_module(self):
                return np
            def to_cpu(self, x):
                return x
            def to_gpu(self, x):
                return x
            def asarray(self, obj, dtype=None):
                return np.asarray(obj, dtype=dtype)
            def zeros(self, shape, dtype=None):
                return np.zeros(shape, dtype=dtype)
            def ones(self, shape, dtype=None):
                return np.ones(shape, dtype=dtype)
            def array(self, obj, dtype=None):
                return np.array(obj, dtype=dtype)
            def copy(self, x):
                return np.copy(x)
        return DummyBackend()

class QuantumFabric:
    """
    Реализация базовых квантовых взаимодействий.
    
    Моделирует квантовую систему из n кубитов с возможностью создания запутанности.
    """
    
    def __init__(self, num_qubits: int, entanglement_strength: float = 1.0, use_gpu: Optional[bool] = None):
        """
        Инициализирует квантовую систему.
        
        Args:
            num_qubits: Количество кубитов в системе
            entanglement_strength: Сила запутанности (0.0 - 1.0)
            use_gpu: Если True - использовать GPU, False - CPU, None - автоопределение
        
        Raises:
            ValueError: Если num_qubits <= 0 или entanglement_strength вне допустимого диапазона
        """
        if num_qubits <= 0:
            raise ValueError(f"num_qubits должен быть > 0, получено {num_qubits}")
        if not 0.0 <= entanglement_strength <= 1.0:
            raise ValueError(f"entanglement_strength должен быть в [0, 1], получено {entanglement_strength}")
        
        self.n = num_qubits
        self.entanglement_strength = entanglement_strength
        self.backend = get_backend(use_gpu=use_gpu)
        self.xp = self.backend.get_array_module()  # xp = numpy или cupy
        self.state = self._initialize_vacuum_state()
        self._normalize_state()
        
    def _initialize_vacuum_state(self):
        """Начальное состояние - вакуум |0>⊗n"""
        # Для OpenCL/Vulkan создаем на CPU, затем конвертируем
        use_gpu_backend = (hasattr(self.backend, 'use_opencl') and self.backend.use_opencl) or \
                         (hasattr(self.backend, 'use_vulkan') and self.backend.use_vulkan)
        if use_gpu_backend:
            state_cpu = np.zeros(2**self.n, dtype=complex)
            state_cpu[0] = 1.0
            return self.backend.to_gpu(state_cpu)
        else:
            state = self.backend.zeros(2**self.n, dtype=complex)
            state[0] = 1.0  # |000...0>
            return state
    
    def _normalize_state(self) -> None:
        """Нормализует квантовое состояние"""
        # Используем linalg из текущего backend (numpy или cupy)
        norm = self.xp.linalg.norm(self.state)
        if norm > 1e-10:  # Избегаем деления на ноль
            self.state = self.state / norm
        else:
            warnings.warn("Состояние близко к нулю, переинициализация")
            self.state = self._initialize_vacuum_state()
    
    def apply_entanglement_operator(self, qubit_pairs: List[Tuple[int, int]], 
                                    use_hadamard: bool = True):
        """
        Применяет операторы запутанности между кубитами.
        
        Для создания запутанности сначала применяет Hadamard на control кубите,
        затем CNOT для создания состояния Белла.
        
        Args:
            qubit_pairs: Список пар индексов кубитов (control, target)
            use_hadamard: Если True, применяет Hadamard перед CNOT (для создания состояния Белла)
        
        Returns:
            Обновленное состояние системы
        """
        for i, j in qubit_pairs:
            if i < 0 or j < 0 or i >= self.n or j >= self.n:
                warnings.warn(f"Пропущена недопустимая пара кубитов: ({i}, {j})")
                continue
            if i == j:
                warnings.warn(f"Пропущена пара с одинаковыми индексами: ({i}, {j})")
                continue
            
            # Для создания запутанности (состояния Белла) нужно:
            # 1. Применить Hadamard на control кубите: |0> -> (|0> + |1>)/√2
            # 2. Применить CNOT: если control=1, то flip target
            if use_hadamard:
                self.apply_hadamard(i)
            
            operator = self._create_entanglement_operator(i, j)
            self.state = operator @ self.state
        
        self._normalize_state()
        return self.state
    
    def _create_entanglement_operator(self, control: int, target: int):
        """
        Создает унитарный оператор запутанности (CNOT).
        
        Оператор создает запутанность между control и target кубитами.
        Использует стандартный CNOT: если control=1, то flip target.
        
        CNOT|xy> = |x, y⊕x>, где ⊕ - XOR
        
        Args:
            control: Индекс control кубита
            target: Индекс target кубита
        
        Returns:
            Унитарная матрица оператора CNOT
        """
        size = 2**self.n
        
        # Для OpenCL/Vulkan создаем оператор на CPU (так как нужен поэлементный доступ)
        # Для CUDA/CPU создаем напрямую
        use_cpu_for_construction = (hasattr(self.backend, 'use_opencl') and self.backend.use_opencl) or \
                                  (hasattr(self.backend, 'use_vulkan') and self.backend.use_vulkan)
        
        if use_cpu_for_construction:
            # Создаем на CPU, затем конвертируем на GPU
            operator_cpu = np.zeros((size, size), dtype=complex)
        else:
            operator = self.backend.zeros((size, size), dtype=complex)
            operator_cpu = None
        
        # CNOT оператор: если control бит = 1, то flip target бит
        # CNOT|xy> = |x, y⊕x>
        for i in range(size):
            control_bit = (i >> control) & 1
            target_bit = (i >> target) & 1
            
            if control_bit == 1:
                # Если control = 1, флипаем target: y -> y⊕1
                j = i ^ (1 << target)  # Флипаем target бит
                if use_cpu_for_construction:
                    operator_cpu[j, i] = 1.0
                else:
                    operator[j, i] = 1.0
            else:
                # Если control = 0, ничего не меняем
                if use_cpu_for_construction:
                    operator_cpu[i, i] = 1.0
                else:
                    operator[i, i] = 1.0
        
        # Применяем параметр силы запутанности
        # Для частичной запутанности используем контролируемое вращение (CRY gate)
        if self.entanglement_strength < 1.0:
            # Пересоздаем оператор с учетом силы запутанности
            if use_cpu_for_construction:
                operator_cpu = np.eye(size, dtype=complex)
            else:
                operator = self.xp.eye(size, dtype=complex)
            
            theta = np.pi * self.entanglement_strength  # Угол вращения
            
            for i in range(size):
                control_bit = (i >> control) & 1
                target_bit = (i >> target) & 1
                
                if control_bit == 1:
                    # Если control = 1, применяем контролируемое вращение Y
                    target_state = i ^ (1 << target)  # Состояние с флипнутым target
                    
                    # Контролируемое вращение Y: CRY(θ)
                    if target_bit == 0:
                        # Исходное состояние |1,0>
                        cos_val = np.cos(theta / 2)
                        sin_val = -np.sin(theta / 2)
                        if use_cpu_for_construction:
                            operator_cpu[i, i] = cos_val
                            operator_cpu[target_state, i] = sin_val
                        else:
                            operator[i, i] = self.xp.cos(theta / 2)
                            operator[target_state, i] = -self.xp.sin(theta / 2)
                    else:
                        # Исходное состояние |1,1>
                        cos_val = np.cos(theta / 2)
                        sin_val = np.sin(theta / 2)
                        if use_cpu_for_construction:
                            operator_cpu[i, i] = cos_val
                            operator_cpu[target_state, i] = sin_val
                        else:
                            operator[i, i] = self.xp.cos(theta / 2)
                            operator[target_state, i] = self.xp.sin(theta / 2)
                else:
                    # Если control = 0, ничего не меняем
                    if use_cpu_for_construction:
                        operator_cpu[i, i] = 1.0
                    else:
                        operator[i, i] = 1.0
        
        # Конвертируем на GPU если нужно
        if use_cpu_for_construction:
            operator = self.backend.asarray(operator_cpu)
        
        # Проверяем унитарность
        if not self._is_unitary(operator, tolerance=1e-6):
            warnings.warn("Оператор запутанности не является строго унитарным, применяется коррекция")
            U, _, Vh = self.xp.linalg.svd(operator)
            operator = U @ Vh
        
        return operator
    
    def apply_hadamard(self, qubit_index: int):
        """
        Применяет ворота Адамара к указанному кубиту.
        
        Hadamard создает суперпозицию: H|0> = (|0> + |1>)/√2
        
        Args:
            qubit_index: Индекс кубита
        
        Returns:
            Обновленное состояние системы
        """
        if qubit_index < 0 or qubit_index >= self.n:
            raise ValueError(f"qubit_index должен быть в [0, {self.n-1}], получено {qubit_index}")
        
        size = 2**self.n
        hadamard_matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        
        # Для OpenCL/Vulkan создаем оператор на CPU
        use_cpu_for_construction = (hasattr(self.backend, 'use_opencl') and self.backend.use_opencl) or \
                                  (hasattr(self.backend, 'use_vulkan') and self.backend.use_vulkan)
        
        if use_cpu_for_construction:
            operator_cpu = np.eye(size, dtype=complex)
        else:
            operator = self.xp.eye(size, dtype=complex)
        
        # Применяем Hadamard к указанному кубиту
        for i in range(size):
            for j in range(size):
                # Проверяем, что все биты кроме qubit_index совпадают
                mask = ~(1 << qubit_index)
                if (i & mask) == (j & mask):
                    bit_i = (i >> qubit_index) & 1
                    bit_j = (j >> qubit_index) & 1
                    if use_cpu_for_construction:
                        operator_cpu[i, j] = hadamard_matrix[bit_i, bit_j]
                    else:
                        operator[i, j] = hadamard_matrix[bit_i, bit_j]
        
        # Конвертируем на GPU если нужно
        if use_cpu_for_construction:
            operator = self.backend.asarray(operator_cpu)
        
        self.state = operator @ self.state
        self._normalize_state()
        return self.state
    
    def _is_unitary(self, operator: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Проверяет, является ли оператор унитарным"""
        return self.xp.allclose(operator @ operator.conj().T, self.xp.eye(len(operator)), atol=tolerance)
    
    def get_state_info(self) -> str:
        """
        Возвращает информацию о текущем состоянии системы.
        
        Returns:
            Строка с описанием состояния
        """
        norm = self.xp.linalg.norm(self.state)
        coherence = float(self.xp.sum(self.xp.abs(self.state)**2))
        return (f"Система из {self.n} кубитов, "
                f"норма состояния: {norm:.6f}, "
                f"когерентность: {coherence:.6f}")
    
    def measure(self, qubit_index: int) -> int:
        """
        Измеряет кубит и коллапсирует состояние.
        
        Args:
            qubit_index: Индекс измеряемого кубита
        
        Returns:
            Результат измерения (0 или 1)
        
        Raises:
            ValueError: Если qubit_index вне допустимого диапазона
        """
        if qubit_index < 0 or qubit_index >= self.n:
            raise ValueError(f"qubit_index должен быть в [0, {self.n-1}], получено {qubit_index}")
        
        # Для OpenCL/Vulkan конвертируем состояние в CPU для индексирования
        use_gpu_backend = (hasattr(self.backend, 'use_opencl') and self.backend.use_opencl) or \
                         (hasattr(self.backend, 'use_vulkan') and self.backend.use_vulkan)
        if use_gpu_backend:
            state_cpu = self.backend.to_cpu(self.state)
        else:
            state_cpu = self.state
        
        # Вычисляем вероятности
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(2**self.n):
            if (i >> qubit_index) & 1:
                prob_1 += float(np.abs(state_cpu[i])**2)
            else:
                prob_0 += float(np.abs(state_cpu[i])**2)
        
        # Коллапс состояния
        result = np.random.choice([0, 1], p=[prob_0, prob_1])
        
        # Проецируем состояние на измеренное значение
        for i in range(2**self.n):
            if ((i >> qubit_index) & 1) != result:
                state_cpu[i] = 0.0
        
        # Конвертируем обратно на GPU если нужно
        if use_gpu_backend:
            self.state = self.backend.to_gpu(state_cpu)
        else:
            self.state = state_cpu
        
        self._normalize_state()
        return result
    
    def get_probability_distribution(self) -> np.ndarray:
        """
        Возвращает распределение вероятностей по всем базисным состояниям.
        
        Returns:
            Массив вероятностей для каждого базисного состояния
        """
        # Конвертируем в CPU для возврата (если GPU)
        probs = self.xp.abs(self.state)**2
        return self.backend.to_cpu(probs)
    
    def get_qubit_probabilities(self, qubit_index: int) -> Tuple[float, float]:
        """
        Возвращает вероятности измерения кубита в состояниях |0> и |1>.
        
        Args:
            qubit_index: Индекс кубита
        
        Returns:
            Кортеж (вероятность |0>, вероятность |1>)
        """
        # Для OpenCL/Vulkan конвертируем в CPU для индексирования
        use_gpu_backend = (hasattr(self.backend, 'use_opencl') and self.backend.use_opencl) or \
                         (hasattr(self.backend, 'use_vulkan') and self.backend.use_vulkan)
        if use_gpu_backend:
            state_cpu = self.backend.to_cpu(self.state)
        else:
            state_cpu = self.state
        
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(2**self.n):
            if (i >> qubit_index) & 1:
                prob_1 += float(np.abs(state_cpu[i])**2)
            else:
                prob_0 += float(np.abs(state_cpu[i])**2)
        
        return (prob_0, prob_1)
    
    def get_entanglement_entropy(self) -> float:
        """
        Вычисляет энтропию фон Неймана как меру запутанности.
        
        Для системы из 2 кубитов вычисляет настоящую энтропию фон Неймана.
        Для больших систем вычисляет среднюю энтропию по всем парам.
        
        Returns:
            Энтропия запутанности (0 = нет запутанности, log2 = максимальная запутанность)
            Нормализованная к диапазону [0, 1]
        """
        if self.n == 1:
            return 0.0
        
        if self.n == 2:
            # Для 2 кубитов вычисляем настоящую энтропию фон Неймана
            # Создаем матрицу плотности для всей системы
            rho = self.xp.outer(self.state, self.state.conj())
            
            # Вычисляем редуцированную матрицу плотности для первого кубита
            # Трассируем по второму кубиту
            rho_reduced = self.backend.zeros((2, 2), dtype=complex)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        rho_reduced[i, j] += rho[i*2 + k, j*2 + k]
            
            # Вычисляем собственные значения
            eigenvals = self.xp.linalg.eigvals(rho_reduced)
            eigenvals = self.xp.real(eigenvals)  # Убираем мнимую часть (должна быть 0)
            eigenvals = eigenvals[eigenvals > 1e-10]  # Убираем нули
            
            # Энтропия фон Неймана: S = -Tr(ρ log2(ρ))
            entropy = -float(self.xp.sum(eigenvals * self.xp.log2(eigenvals + 1e-10)))
            
            # Нормализуем к [0, 1] (максимальная энтропия для 2 кубитов = 1)
            return float(entropy)
        else:
            # Для больших систем вычисляем среднюю энтропию по всем парам
            # Используем правильную редуцированную матрицу плотности
            entropies = []
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    # Создаем редуцированную матрицу плотности для пары кубитов (i, j)
                    # Трассируем по всем остальным кубитам
                    rho_ij = self.backend.zeros((4, 4), dtype=complex)
                    
                    # Перебираем все базисные состояния системы
                    for k in range(2**self.n):
                        bit_i = (k >> i) & 1
                        bit_j = (k >> j) & 1
                        
                        # Индекс в подпространстве пары (i, j)
                        idx_ij = bit_i * 2 + bit_j
                        
                        # Суммируем по всем остальным кубитам
                        for l in range(2**self.n):
                            # Проверяем, что биты i и j совпадают
                            if ((l >> i) & 1) == bit_i and ((l >> j) & 1) == bit_j:
                                bit_i_l = (l >> i) & 1
                                bit_j_l = (l >> j) & 1
                                idx_ij_l = bit_i_l * 2 + bit_j_l
                                
                                # Проверяем, что все остальные биты совпадают
                                mask = ~((1 << i) | (1 << j))
                                if (k & mask) == (l & mask):
                                    # Для OpenCL конвертируем в CPU
                                    state_k = float(self.backend.to_cpu(self.state)[k]) if hasattr(self.backend, 'use_opencl') and self.backend.use_opencl else self.state[k]
                                    state_l = float(self.backend.to_cpu(self.state)[l]) if hasattr(self.backend, 'use_opencl') and self.backend.use_opencl else self.state[l]
                                    rho_ij[idx_ij, idx_ij_l] += state_k * np.conj(state_l)
                    
                    # Вычисляем редуцированную матрицу плотности для кубита i
                    # (трассируем по кубиту j)
                    rho_i = self.backend.zeros((2, 2), dtype=complex)
                    for a in range(2):
                        for b in range(2):
                            for j_state in range(2):
                                rho_i[a, b] += rho_ij[a * 2 + j_state, b * 2 + j_state]
                    
                    # Вычисляем энтропию
                    eigenvals = self.xp.linalg.eigvals(rho_i)
                    eigenvals = self.xp.real(eigenvals)
                    eigenvals = eigenvals[eigenvals > 1e-10]
                    
                    if len(eigenvals) > 0 and float(self.xp.sum(eigenvals)) > 1e-10:
                        # Нормализуем собственные значения
                        eigenvals = eigenvals / self.xp.sum(eigenvals)
                        entropy = -float(self.xp.sum(eigenvals * self.xp.log2(eigenvals + 1e-10)))
                        # Нормализуем к [0, 1] (максимальная энтропия для одного кубита = 1)
                        entropy = min(entropy, 1.0)
                        entropies.append(entropy)
            
            if len(entropies) > 0:
                return float(self.xp.mean(self.xp.array(entropies)))
            else:
                return 0.0
    
    def get_coherence(self) -> float:
        """
        Вычисляет меру когерентности состояния.
        
        Returns:
            Мера когерентности (сумма квадратов модулей амплитуд)
        """
        return float(self.xp.sum(self.xp.abs(self.state)**2))
    
    def get_state_vector(self):
        """
        Возвращает вектор состояния (копию).
        
        Returns:
            Копия вектора состояния (конвертируется в CPU если GPU)
        """
        state_copy = self.backend.copy(self.state)
        return self.backend.to_cpu(state_copy)
    
    def collect_measurement_statistics(self, qubit_index: int, num_measurements: int) -> dict:
        """
        Собирает статистику множественных измерений кубита.
        
        Args:
            qubit_index: Индекс измеряемого кубита
            num_measurements: Количество измерений
        
        Returns:
            Словарь со статистикой: {'count_0': int, 'count_1': int, 'prob_0': float, 'prob_1': float}
        """
        # Сохраняем исходное состояние
        original_state = self.backend.copy(self.state)
        
        count_0 = 0
        count_1 = 0
        
        for _ in range(num_measurements):
            # Восстанавливаем состояние перед каждым измерением
            self.state = self.backend.copy(original_state)
            result = self.measure(qubit_index)
            if result == 0:
                count_0 += 1
            else:
                count_1 += 1
        
        # Восстанавливаем исходное состояние
        self.state = original_state
        
        return {
            'count_0': count_0,
            'count_1': count_1,
            'prob_0': count_0 / num_measurements,
            'prob_1': count_1 / num_measurements,
            'num_measurements': num_measurements
        }
