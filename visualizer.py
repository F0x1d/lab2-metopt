"""
Модуль для визуализации результатов оптимизации.

Создает графики:
- Исходная функция
- Точки, в которых вычислялась функция
- Ломаная линия, соединяющая точки
- Вспомогательные функции (нижние оценки)
- Найденный минимум
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Tuple


class Visualizer:
    """
    Класс для визуализации процесса и результатов оптимизации.
    """

    def __init__(self, optimizer):
        """
        Инициализация визуализатора.

        Args:
            optimizer: Объект GlobalOptimizer после выполнения оптимизации
        """
        self.optimizer = optimizer

    def plot_results(self, save_path: str = None, show_auxiliary: bool = True):
        """
        Построение итогового графика с результатами оптимизации.

        Args:
            save_path: Путь для сохранения графика (если None, график только показывается)
            show_auxiliary: Показывать ли вспомогательные функции
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # 1. График исходной функции
        # Создаем плотную сетку для красивого графика
        x_dense = np.linspace(self.optimizer.a, self.optimizer.b, 1000)
        y_dense = [self.optimizer.function(x) for x in x_dense]

        ax.plot(x_dense, y_dense, 'b-', linewidth=2, label='Исходная функция f(x)', zorder=1)

        # 2. Ломаная линия (соединяет все вычисленные точки)
        x_sorted, f_sorted = self.optimizer.get_piecewise_linear_approximation()
        ax.plot(x_sorted, f_sorted, 'g--', linewidth=1.5,
                label='Ломаная линия (вычисленные точки)', zorder=2)

        # 3. Точки, в которых вычислялась функция
        ax.scatter(self.optimizer.x_points, self.optimizer.f_points,
                   color='orange', s=50, zorder=3, label='Вычисленные точки')

        # 4. Найденный минимум
        min_index = np.argmin(self.optimizer.f_points)
        x_min = self.optimizer.x_points[min_index]
        f_min = self.optimizer.f_points[min_index]

        ax.scatter([x_min], [f_min], color='red', s=200, marker='*',
                   zorder=4, label=f'Найденный минимум: ({x_min:.4f}, {f_min:.4f})')

        # 5. Вспомогательные функции (последняя итерация)
        if show_auxiliary and len(self.optimizer.history) > 0:
            self._plot_auxiliary_functions(ax)

        # Настройка графика
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('f(x)', fontsize=12)
        ax.set_title(f'Поиск глобального минимума: {self.optimizer.function_str}',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Сохранение или показ
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен в: {save_path}")

        plt.show()

    def _plot_auxiliary_functions(self, ax):
        """
        Отрисовка вспомогательных функций (нижних оценок) на последней итерации.

        Вспомогательные функции строятся для каждого отрезка между соседними точками.
        Это конусообразные функции, которые аппроксимируют исходную функцию снизу.

        Args:
            ax: Объект matplotlib axes
        """
        if len(self.optimizer.history) < 2:
            return

        # Берем последнюю итерацию
        last_state = self.optimizer.history[-1]
        L = last_state.get('L', 1.0)

        # Сортируем точки
        sorted_indices = np.argsort(last_state['x_points'])
        x_sorted = [last_state['x_points'][i] for i in sorted_indices]
        f_sorted = [last_state['f_points'][i] for i in sorted_indices]

        # Для каждого отрезка строим вспомогательную функцию
        for i in range(len(x_sorted) - 1):
            x_left = x_sorted[i]
            f_left = f_sorted[i]
            x_right = x_sorted[i + 1]
            f_right = f_sorted[i + 1]

            # Создаем сетку точек на отрезке
            x_segment = np.linspace(x_left, x_right, 50)

            # Вспомогательная функция: R(x) = max(левый конус, правый конус)
            # Левый конус: f_left - L*(x - x_left)
            # Правый конус: f_right - L*(x_right - x)
            left_cone = f_left - L * (x_segment - x_left)
            right_cone = f_right - L * (x_right - x_segment)
            R = np.maximum(left_cone, right_cone)

            # Отрисовываем вспомогательную функцию
            if i == 0:  # Добавляем label только один раз
                ax.plot(x_segment, R, 'r-', alpha=0.5, linewidth=1,
                        label='Вспомогательные функции', zorder=0)
            else:
                ax.plot(x_segment, R, 'r-', alpha=0.5, linewidth=1, zorder=0)

    def plot_convergence(self, save_path: str = None):
        """
        График сходимости: показывает как улучшается найденный минимум с итерациями.

        Args:
            save_path: Путь для сохранения графика
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Для каждой итерации находим лучший (минимальный) найденный результат
        iterations = []
        best_values = []

        for state in self.optimizer.history:
            iterations.append(state['iteration'])
            best_values.append(min(state['f_points']))

        ax.plot(iterations, best_values, 'b-o', linewidth=2, markersize=5)
        ax.set_xlabel('Итерация', fontsize=12)
        ax.set_ylabel('Лучшее найденное значение f(x)', fontsize=12)
        ax.set_title('График сходимости алгоритма', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сходимости сохранен в: {save_path}")

        plt.show()

    def create_animation_frames(self, output_dir: str = 'frames', max_frames: int = 20):
        """
        Создает последовательность кадров для анимации процесса оптимизации.
        Полезно для презентаций и понимания работы алгоритма.

        Args:
            output_dir: Директория для сохранения кадров
            max_frames: Максимальное число кадров (если итераций больше, берем равномерную выборку)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Выбираем итерации для отображения
        total_iterations = len(self.optimizer.history)
        if total_iterations <= max_frames:
            selected_iterations = range(total_iterations)
        else:
            # Равномерная выборка итераций
            step = total_iterations / max_frames
            selected_iterations = [int(i * step) for i in range(max_frames)]

        # Создаем кадры
        x_dense = np.linspace(self.optimizer.a, self.optimizer.b, 1000)
        y_dense = [self.optimizer.function(x) for x in x_dense]

        for idx, iter_idx in enumerate(selected_iterations):
            state = self.optimizer.history[iter_idx]

            fig, ax = plt.subplots(figsize=(12, 8))

            # Исходная функция
            ax.plot(x_dense, y_dense, 'b-', linewidth=2, label='f(x)')

            # Вычисленные точки на текущей итерации
            ax.scatter(state['x_points'], state['f_points'],
                       color='orange', s=50, zorder=3)

            # Ломаная линия
            x_sorted = sorted(zip(state['x_points'], state['f_points']))
            x_s = [p[0] for p in x_sorted]
            f_s = [p[1] for p in x_sorted]
            ax.plot(x_s, f_s, 'g--', linewidth=1.5)

            # Текущий лучший минимум
            min_idx = np.argmin(state['f_points'])
            ax.scatter([state['x_points'][min_idx]], [state['f_points'][min_idx]],
                       color='red', s=200, marker='*', zorder=4)

            ax.set_title(f"Итерация {state['iteration']}", fontsize=14)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('f(x)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/frame_{idx:03d}.png", dpi=150)
            plt.close()

        print(f"Создано {len(selected_iterations)} кадров в директории '{output_dir}'")
