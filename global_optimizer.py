import numpy as np
import time
from typing import Callable, Tuple, List, Dict


class GlobalOptimizer:

    def __init__(self, function_str: str, a: float, b: float, eps: float = 0.01):
        self.function_str = function_str
        self.a = a
        self.b = b
        self.eps = eps
        self.function = self._create_function(function_str)

        # Статистика
        self.iterations = 0
        self.time_spent = 0.0

        # История вычислений
        self.x_points = []
        self.f_points = []
        self.history = []

    def _create_function(self, func_str: str) -> Callable:
        safe_dict = {
            'np': np,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'pi': np.pi,
        }

        def f(x):
            safe_dict['x'] = x
            return eval(func_str, {"__builtins__": {}}, safe_dict)

        return f

    def _estimate_lipschitz_constant(self, r: float = 2.0) -> float:
        if len(self.x_points) < 2:
            return 1.0

        L_max = 0.0
        for i in range(len(self.x_points)):
            for j in range(i + 1, len(self.x_points)):
                dx = abs(self.x_points[i] - self.x_points[j])
                if dx > 1e-10:
                    df = abs(self.f_points[i] - self.f_points[j])
                    L = df / dx
                    L_max = max(L_max, L)

        return r * L_max if L_max > 1e-10 else 1.0

    def _compute_auxiliary_minimum(self, x_left: float, f_left: float,
                                   x_right: float, f_right: float,
                                   L: float) -> Tuple[float, float]:
        x_min = (x_left + x_right) / 2.0 - (f_right - f_left) / (2.0 * L)

        x_min = max(x_left, min(x_right, x_min))

        R_min = (f_left + f_right) / 2.0 - L * (x_right - x_left) / 2.0

        return x_min, R_min

    def optimize(self, max_iterations: int = 1000) -> Dict:
        start_time = time.time()

        self.x_points = [self.a, self.b]
        self.f_points = [self.function(self.a), self.function(self.b)]
        self.iterations = 2  # Уже сделали 2 вычисления функции

        self.history.append({
            'x_points': self.x_points.copy(),
            'f_points': self.f_points.copy(),
            'iteration': 0
        })

        for iteration in range(max_iterations):
            L = self._estimate_lipschitz_constant()

            candidates = []

            sorted_indices = np.argsort(self.x_points)
            x_sorted = [self.x_points[i] for i in sorted_indices]
            f_sorted = [self.f_points[i] for i in sorted_indices]

            for i in range(len(x_sorted) - 1):
                x_left = x_sorted[i]
                f_left = f_sorted[i]
                x_right = x_sorted[i + 1]
                f_right = f_sorted[i + 1]

                x_min, R_min = self._compute_auxiliary_minimum(
                    x_left, f_left, x_right, f_right, L
                )

                candidates.append({
                    'x': x_min,
                    'R': R_min,
                    'interval': (x_left, x_right)
                })

            best_candidate = min(candidates, key=lambda c: c['R'])
            x_new = best_candidate['x']

            f_new = self.function(x_new)
            self.iterations += 1

            self.x_points.append(x_new)
            self.f_points.append(f_new)

            self.history.append({
                'x_points': self.x_points.copy(),
                'f_points': self.f_points.copy(),
                'iteration': iteration + 1,
                'L': L,
                'best_interval': best_candidate['interval']
            })

            f_min_current = min(self.f_points)
            R_min = best_candidate['R']

            if abs(f_min_current - R_min) < self.eps:
                break

        self.time_spent = time.time() - start_time

        min_index = np.argmin(self.f_points)
        x_min = self.x_points[min_index]
        f_min = self.f_points[min_index]

        return {
            'x_min': x_min,
            'f_min': f_min,
            'iterations': self.iterations,
            'time_spent': self.time_spent,
            'success': iteration < max_iterations - 1
        }

    def get_piecewise_linear_approximation(self) -> Tuple[List[float], List[float]]:
        sorted_indices = np.argsort(self.x_points)
        x_sorted = [self.x_points[i] for i in sorted_indices]
        f_sorted = [self.f_points[i] for i in sorted_indices]

        return x_sorted, f_sorted
