import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cmath

# Создаем директории для сохранения результатов
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/data", exist_ok=True)

# Константы с новыми значениями
TAU = 0.003  # время релаксации
K = 0.005    # коэффициент теплопроводности Фурье
L = 1.0      # размер области [-l, l]

def calculate_coefficients(omega):
    """
    Вычисляет все коэффициенты, зависящие от omega, включая возможные комплексные значения
    
    Args:
        omega: Значение омеги (частоты в области Фурье)
        
    Returns:
        Словарь с вычисленными коэффициентами
    """
    try:
        # Базовые коэффициенты
        a = 1/TAU
        b = K * omega**2 / TAU
        c = L * omega**2 / TAU
        
        # Промежуточные вычисления для уравнения вида z^3 + pz + q = 0
        p = -a**2 / 3
        q = 2 * (a/3)**3 - a*b/3 + c
        
        # Дискриминант
        delta = (p/3)**3 + (q/2)**2
        
        # Вычисление альфа и бета
        if delta >= 0:
            # Один вещественный и два комплексных корня
            alpha = complex(-q/2 + np.sqrt(delta))**(1/3)
            beta = complex(-q/2 - np.sqrt(delta))**(1/3)
        else:
            # Три вещественных корня - используем тригонометрическую формулу
            r = np.sqrt(-p**3 / 27)
            theta = np.arccos(-q / (2*r)) / 3
            
            # Три корня уравнения z^3 + pz + q = 0
            z1 = 2 * np.sqrt(-p/3) * np.cos(theta)
            z2 = 2 * np.sqrt(-p/3) * np.cos(theta + 2*np.pi/3)
            z3 = 2 * np.sqrt(-p/3) * np.cos(theta + 4*np.pi/3)
            
            # Выбираем корни для альфа и бета
            alpha = z1
            beta = z3
        
        # Вычисляем A и B
        A = (alpha + beta) / 2
        
        # Для мнимой части B используем разность между альфа и бета
        if isinstance(alpha, complex) or isinstance(beta, complex):
            B = abs(1j * np.sqrt(3) * (alpha - beta) / 2)
        else:
            B = abs(np.sqrt(3) * (alpha - beta) / 2)
        
        # Вычисляем другие коэффициенты
        C = a/3
        D = a*(a-3)/9
        
        # Финальные коэффициенты
        denominator = 9*A**2 + B**2
        if abs(denominator) < 1e-15:
            denominator = 1e-15  # Предотвращение деления на ноль
            
        E = (4*A**2 + 2*A*C + D) / denominator
        F = 1 - E
        G = (-3*A**3 - A*B**2 + (3*A**2 + B**2)*C - 3*A*D) / denominator
        
        # Коэффициенты m1 и m2
        m1 = -2*A + a/3
        m2 = A + a/3
        
        return {
            'a': a, 
            'b': b, 
            'c': c,
            'p': p,
            'q': q,
            'delta': delta,
            'alpha': alpha,
            'beta': beta,
            'A': A,
            'B': B,
            'C': C, 
            'D': D,
            'E': E, 
            'F': F, 
            'G': G,
            'm1': m1, 
            'm2': m2
        }
    except Exception as e:
        print(f"Error calculating coefficients for omega={omega}: {e}")
        return None

def calculate_T(omega, t, coeffs=None):
    """
    Вычисляет значение функции T(omega) для заданного времени t
    
    Args:
        omega: Значение омеги (частоты в области Фурье)
        t: Время
        coeffs: Предварительно рассчитанные коэффициенты (опционально)
        
    Returns:
        Значение функции T(omega, t) (комплексное)
    """
    try:
        # Получаем коэффициенты, если они не были переданы
        if coeffs is None:
            coeffs = calculate_coefficients(omega)
            if coeffs is None:
                return complex(0, 0)
        
        # Извлекаем нужные коэффициенты
        A = coeffs['A']
        B = coeffs['B']
        E = coeffs['E']
        F = coeffs['F']
        G = coeffs['G']
        m1 = coeffs['m1']
        m2 = coeffs['m2']
        
        # Вычисляем функцию T(omega)
        term1 = np.exp(-m1 * t) * E
        
        # Обрабатываем случай, когда B почти равно нулю
        if abs(B) < 1e-10:
            term2 = np.exp(-m2 * t) * F
        else:
            term2 = np.exp(-m2 * t) * (F * np.cos(B * t) + G * np.sin(B * t) / B)
        
        return term1 + term2
        
    except Exception as e:
        print(f"Error calculating T for omega={omega}, t={t}: {e}")
        return complex(0, 0)

def analyze_coefficients(omega_range, n_points=1000):
    """
    Анализирует коэффициенты в зависимости от omega
    
    Args:
        omega_range: Диапазон значений omega (min, max)
        n_points: Количество точек для анализа
    
    Returns:
        Диапазоны omega, где все коэффициенты вещественные
    """
    # Используем логарифмическую шкалу для omega, чтобы лучше покрыть большой диапазон
    omega_values = np.logspace(np.log10(max(0.1, omega_range[0])), np.log10(omega_range[1]), n_points)
    
    # Добавляем omega=0 отдельно, если диапазон начинается с 0
    if omega_range[0] == 0:
        omega_values = np.concatenate(([0], omega_values))
    
    # Инициализируем массивы для хранения значений коэффициентов
    coef_names = ['alpha', 'beta', 'delta', 'A', 'B', 'E', 'F', 'G', 'm1', 'm2']
    coef_values = {name: np.zeros(len(omega_values), dtype=complex) for name in coef_names}
    
    # Массив для отслеживания вещественности всех коэффициентов
    all_real = np.ones(len(omega_values), dtype=bool)
    
    print(f"Анализ коэффициентов для {len(omega_values)} значений omega...")
    start_time = time.time()
    
    # Вычисляем коэффициенты для каждого значения omega
    for i, omega in enumerate(omega_values):
        coeffs = calculate_coefficients(omega)
        if coeffs is None:
            all_real[i] = False
            continue
            
        for name in coef_names:
            val = coeffs[name]
            coef_values[name][i] = val
            
            # Проверяем, является ли коэффициент вещественным
            if isinstance(val, complex) and abs(val.imag) > 1e-10:
                all_real[i] = False
        
        # Выводим прогресс
        if (i+1) % (n_points//10) == 0 or i == len(omega_values) - 1:
            elapsed = time.time() - start_time
            print(f"Обработано {i+1}/{len(omega_values)} значений omega ({elapsed:.2f} сек)")
    
    # Строим графики для выбранных коэффициентов (alpha, beta, delta)
    for name in ['alpha', 'beta', 'delta']:
        plt.figure(figsize=(15, 10))
        
        real_part = np.real(coef_values[name])
        imag_part = np.imag(coef_values[name])
        magnitude = np.abs(coef_values[name])
        
        plt.subplot(3, 1, 1)
        plt.semilogx(omega_values, real_part, 'b-')
        plt.title(f'Вещественная часть {name}(ω)')
        plt.xlabel('ω (частота)')
        plt.ylabel(f'Re[{name}(ω)]')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.semilogx(omega_values, imag_part, 'r-')
        plt.title(f'Мнимая часть {name}(ω)')
        plt.xlabel('ω (частота)')
        plt.ylabel(f'Im[{name}(ω)]')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.semilogx(omega_values, magnitude, 'g-')
        plt.title(f'Модуль |{name}(ω)|')
        plt.xlabel('ω (частота)')
        plt.ylabel(f'|{name}(ω)|')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"results/figures/coefficient_{name}_analysis.png", dpi=300)
        plt.close()
    
    # Строим графики для основных коэффициентов (A, B, E, F, G, m1, m2)
    for name in ['A', 'B', 'E', 'F', 'G', 'm1', 'm2']:
        plt.figure(figsize=(15, 10))
        
        real_part = np.real(coef_values[name])
        imag_part = np.imag(coef_values[name])
        magnitude = np.abs(coef_values[name])
        
        plt.subplot(3, 1, 1)
        plt.semilogx(omega_values, real_part, 'b-')
        plt.title(f'Вещественная часть {name}(ω)')
        plt.xlabel('ω (частота)')
        plt.ylabel(f'Re[{name}(ω)]')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.semilogx(omega_values, imag_part, 'r-')
        plt.title(f'Мнимая часть {name}(ω)')
        plt.xlabel('ω (частота)')
        plt.ylabel(f'Im[{name}(ω)]')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.semilogx(omega_values, magnitude, 'g-')
        plt.title(f'Модуль |{name}(ω)|')
        plt.xlabel('ω (частота)')
        plt.ylabel(f'|{name}(ω)|')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"results/figures/coefficient_{name}_analysis.png", dpi=300)
        plt.close()
    
    # Дополнительно строим график дельты для анализа корней
    plt.figure(figsize=(12, 6))
    plt.semilogx(omega_values, np.real(coef_values['delta']), 'b-')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Зависимость дискриминанта delta от частоты ω')
    plt.xlabel('ω (частота)')
    plt.ylabel('delta(ω)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/figures/delta_analysis.png", dpi=300)
    plt.close()
    
    # Находим диапазоны omega, где все коэффициенты вещественные
    real_ranges = []
    start_idx = None
    
    for i in range(len(all_real)):
        if all_real[i] and start_idx is None:
            start_idx = i
        elif not all_real[i] and start_idx is not None:
            real_ranges.append((omega_values[start_idx], omega_values[i-1]))
            start_idx = None
    
    # Обрабатываем последний диапазон, если он есть
    if start_idx is not None:
        real_ranges.append((omega_values[start_idx], omega_values[-1]))
    
    print("\nДиапазоны omega, где все коэффициенты вещественные:")
    for i, (start, end) in enumerate(real_ranges):
        print(f"Диапазон {i+1}: [{start}, {end}]")
    
    # Сохраняем данные
    np.savez("results/data/coefficient_analysis.npz",
             omega_values=omega_values,
             coef_values=coef_values,
             all_real=all_real)
    
    return omega_values, all_real, real_ranges

def plot_T_for_real_coefficients(omega_values, all_real, t_values):
    """
    Строит T(omega) для указанных значений t, используя только omega с вещественными коэффициентами
    
    Args:
        omega_values: Массив значений omega
        all_real: Массив булевых значений, указывающий, где все коэффициенты вещественные
        t_values: Список значений времени t для построения графиков
    """
    # Фильтруем только те omega, где все коэффициенты вещественные
    real_omegas = omega_values[all_real]
    
    if len(real_omegas) == 0:
        print("Не найдено значений omega, где все коэффициенты вещественные.")
        return
    
    print(f"\nПостроение T(omega) для {len(real_omegas)} значений omega с вещественными коэффициентами...")
    
    # Создаем массив для хранения значений T
    T_values = np.zeros((len(t_values), len(real_omegas)), dtype=complex)
    
    start_time = time.time()
    
    # Вычисляем T для каждого значения omega и t
    for i, t in enumerate(t_values):
        for j, omega in enumerate(real_omegas):
            # Сначала вычисляем коэффициенты
            coeffs = calculate_coefficients(omega)
            if coeffs is not None:
                T_values[i, j] = calculate_T(omega, t, coeffs)
        
        # Выводим прогресс
        elapsed = time.time() - start_time
        print(f"Обработано {i+1}/{len(t_values)} значений t ({elapsed:.2f} сек)")
    
    # Строим графики для каждой части T в отдельных фигурах
    plt.figure(figsize=(15, 10))
    
    # Вещественная часть
    plt.subplot(3, 1, 1)
    for i, t in enumerate(t_values):
        plt.semilogx(real_omegas, np.real(T_values[i, :]), label=f't={t}')
    plt.title('Вещественная часть T(ω)')
    plt.xlabel('ω (частота)')
    plt.ylabel('Re[T(ω)]')
    plt.grid(True)
    plt.legend()
    
    # Мнимая часть
    plt.subplot(3, 1, 2)
    for i, t in enumerate(t_values):
        plt.semilogx(real_omegas, np.imag(T_values[i, :]), label=f't={t}')
    plt.title('Мнимая часть T(ω)')
    plt.xlabel('ω (частота)')
    plt.ylabel('Im[T(ω)]')
    plt.grid(True)
    plt.legend()
    
    # Модуль
    plt.subplot(3, 1, 3)
    for i, t in enumerate(t_values):
        plt.semilogx(real_omegas, np.abs(T_values[i, :]), label=f't={t}')
    plt.title('Модуль |T(ω)|')
    plt.xlabel('ω (частота)')
    plt.ylabel('|T(ω)|')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/figures/T_omega_real_coefficients.png", dpi=300)
    plt.close()
    
    # Строим также один общий график для всех t (только вещественная часть)
    plt.figure(figsize=(12, 8))
    for i, t in enumerate(t_values):
        plt.semilogx(real_omegas, np.real(T_values[i, :]), label=f't={t}')
    plt.title('Вещественная часть T(ω) для разных значений t')
    plt.xlabel('ω (частота)')
    plt.ylabel('Re[T(ω)]')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/T_omega_real_all_t.png", dpi=300)
    plt.close()
    
    # Сохраняем данные
    np.savez("results/data/T_omega_real_coefficients.npz",
             omega_values=real_omegas,
             t_values=t_values,
             T_values=T_values)

def plot_all_T_values(omega_range, t_values, n_points=1000):
    """
    Строит T(omega) для всех значений omega, не ограничиваясь только теми, где коэффициенты вещественные
    
    Args:
        omega_range: Диапазон значений omega (min, max)
        t_values: Список значений времени t
        n_points: Количество точек для построения графика
    """
    # Создаем массив значений omega
    omega_values = np.logspace(np.log10(max(0.1, omega_range[0])), np.log10(omega_range[1]), n_points)
    
    # Добавляем omega=0 отдельно, если диапазон начинается с 0
    if omega_range[0] == 0:
        omega_values = np.concatenate(([0], omega_values))
    
    # Массив для хранения значений T
    T_values = np.zeros((len(t_values), len(omega_values)), dtype=complex)
    
    print(f"\nПостроение T(omega) для всего диапазона {len(omega_values)} значений omega...")
    start_time = time.time()
    
    # Вычисляем T для каждого значения omega и t
    for i, t in enumerate(t_values):
        for j, omega in enumerate(omega_values):
            T_values[i, j] = calculate_T(omega, t)
        
        # Выводим прогресс
        elapsed = time.time() - start_time
        print(f"Обработано {i+1}/{len(t_values)} значений t ({elapsed:.2f} сек)")
    
    # Строим общий график для всех t (вещественная часть)
    plt.figure(figsize=(12, 8))
    for i, t in enumerate(t_values):
        plt.semilogx(omega_values, np.real(T_values[i, :]), label=f't={t}')
    plt.title('Вещественная часть T(ω) для разных значений t (все omega)')
    plt.xlabel('ω (частота)')
    plt.ylabel('Re[T(ω)]')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/T_omega_all_t.png", dpi=300)
    plt.close()
    
    # Строим также общий график для модуля T
    plt.figure(figsize=(12, 8))
    for i, t in enumerate(t_values):
        plt.semilogx(omega_values, np.abs(T_values[i, :]), label=f't={t}')
    plt.title('Модуль |T(ω)| для разных значений t (все omega)')
    plt.xlabel('ω (частота)')
    plt.ylabel('|T(ω)|')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/T_omega_abs_all_t.png", dpi=300)
    plt.close()
    
    # Сохраняем данные
    np.savez("results/data/T_omega_all.npz",
             omega_values=omega_values,
             t_values=t_values,
             T_values=T_values)

def main():
    """Основная функция скрипта"""
    print("Начало выполнения скрипта для анализа коэффициентов и T(omega)")
    print(f"Константы: TAU={TAU}, K={K}, L={L}")
    
    # Параметры для анализа
    omega_range = (0, 100000)
    n_points = 1000
    t_values = [0, 0.001, 0.002, 0.003]
    
    # Анализируем коэффициенты
    omega_values, all_real, real_ranges = analyze_coefficients(omega_range, n_points)
    
    # Строим T(omega) для выбранных значений t и omega, где все коэффициенты вещественные
    plot_T_for_real_coefficients(omega_values, all_real, t_values)
    
    # Строим T(omega) для всех значений omega
    plot_all_T_values(omega_range, t_values, n_points=1000)
    
    print("Выполнение скрипта завершено")

if __name__ == "__main__":
    main()
