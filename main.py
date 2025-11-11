from global_optimizer import GlobalOptimizer
from visualizer import Visualizer
import os


def main():
    """
    Основная функция для интерактивного ввода и оптимизации.
    """
    print("=" * 80)
    print("ПРОГРАММА ПОИСКА ГЛОБАЛЬНОГО МИНИМУМА ФУНКЦИИ")
    print("Метод Пиявского-Шуберта для липшицевых функций")
    print("=" * 80)

    # Ввод функции
    print("\nВведите функцию одной переменной x.")
    print("Доступные функции: sin, cos, tan, exp, log, sqrt, abs, np.pi")
    print("Примеры:")
    print("  - x**2 + np.sin(x)")
    print("  - x*np.sin(3*x) + np.cos(5*x)")
    print("  - 10 + x**2 - 10*np.cos(2*np.pi*x)  [функция Растригина]")
    print("  - -20*np.exp(-0.2*np.abs(x)) - np.exp(np.cos(2*np.pi*x)) + 20 + np.e  [функция Экли]")

    function_str = input("\nВведите функцию f(x) = ").strip()

    # Ввод границ отрезка
    print("\nВведите границы отрезка [a, b]:")
    try:
        a = float(input("  a = ").strip())
        b = float(input("  b = ").strip())

        if a >= b:
            print("ОШИБКА: Левая граница должна быть меньше правой!")
            return
    except ValueError:
        print("ОШИБКА: Границы должны быть числами!")
        return

    # Ввод точности
    print("\nВведите точность вычисления eps (например, 0.01):")
    try:
        eps = float(input("  eps = ").strip())
        if eps <= 0:
            print("ОШИБКА: Точность должна быть положительным числом!")
            return
    except ValueError:
        print("ОШИБКА: Точность должна быть числом!")
        return

    # Вывод введенных данных
    print("\n" + "=" * 80)
    print("ПАРАМЕТРЫ ОПТИМИЗАЦИИ:")
    print("=" * 80)
    print(f"Функция: f(x) = {function_str}")
    print(f"Отрезок: [{a}, {b}]")
    print(f"Точность: {eps}")

    # Создание оптимизатора
    try:
        optimizer = GlobalOptimizer(function_str, a, b, eps)
    except Exception as e:
        print(f"\nОШИБКА при создании функции: {e}")
        print("Проверьте правильность ввода функции!")
        return

    # Запуск оптимизации
    print("\n" + "-" * 80)
    print("Запуск оптимизации...")
    print("-" * 80)

    try:
        result = optimizer.optimize(max_iterations=1000)
    except Exception as e:
        print(f"\nОШИБКА при оптимизации: {e}")
        return

    # Вывод результатов
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
    print("=" * 80)
    print(f"\nНайденное приближенное значение:")
    print(f"  x* ≈ {result['x_min']:.10f}")
    print(f"  f(x*) ≈ {result['f_min']:.10f}")
    print(f"\nСтатистика:")
    print(f"  Число итераций: {result['iterations']}")
    print(f"  Затраченное время: {result['time_spent']:.6f} секунд")
    print(f"  Статус: {'✓ Успешно (достигнута заданная точность)' if result['success'] else '⚠ Достигнут лимит итераций'}")

    # Визуализация
    print("\n" + "-" * 80)
    print("Создание визуализации...")
    print("-" * 80)

    # Создаем директорию для результатов
    os.makedirs('results', exist_ok=True)

    # Генерируем имя файла на основе функции (безопасное)
    safe_filename = "".join(c if c.isalnum() else "_" for c in function_str[:30])

    result_path = f'results/{safe_filename}_result.png'
    convergence_path = f'results/{safe_filename}_convergence.png'

    vis = Visualizer(optimizer)

    print("\n1. Создание основного графика...")
    vis.plot_results(save_path=result_path, show_auxiliary=True)

    print("\n2. Создание графика сходимости...")
    vis.plot_convergence(save_path=convergence_path)

    print("\n" + "=" * 80)
    print("ВИЗУАЛИЗАЦИЯ СОХРАНЕНА:")
    print("=" * 80)
    print(f"  - {result_path}")
    print(f"  - {convergence_path}")

    print("\n" + "=" * 80)
    print("РАБОТА ПРОГРАММЫ ЗАВЕРШЕНА")
    print("=" * 80)


if __name__ == "__main__":
    main()
