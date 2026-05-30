import numpy as np
import matplotlib.pyplot as plt

def create_mock_data():
    t = np.linspace(0, 1.0, 100)
    g = 9.81
    height = 1.0
    t_impact = np.sqrt(2 * height / g)
    
    y_hard = np.zeros_like(t)
    y_soft = np.zeros_like(t)
    
    e_hard = 0.6
    e_soft = 0.2
    
    for i, time in enumerate(t):
        if time < t_impact:
            y = height - 0.5 * g * time**2
            y_hard[i] = y
            y_soft[i] = y
        else:
            t_after = time - t_impact
            v_impact = g * t_impact
            y_hard_val = (v_impact * e_hard) * t_after - 0.5 * g * t_after**2
            y_hard[i] = max(y_hard_val, 0)
            y_soft_val = (v_impact * e_soft) * t_after - 0.5 * g * t_after**2
            y_soft[i] = max(y_soft_val, 0)
            
    return t, y_hard, y_soft

def calculate_velocity_acceleration(t, y):
    dt = t[1] - t[0]
    v = np.gradient(y, dt)
    a = np.gradient(v, dt)
    return v, a

def main():
    t, y_hard, y_soft = create_mock_data()
    v_hard, a_hard = calculate_velocity_acceleration(t, y_hard)
    v_soft, a_soft = calculate_velocity_acceleration(t, y_soft)
    
    # 한글 폰트 설정 (Windows 기준 맑은 고딕)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 5))
    plt.plot(t, y_hard, label='완충재 없음 (단단한 바닥)', color='red')
    plt.plot(t, y_soft, label='완충재 있음 (충격 흡수)', color='blue', linestyle='--')
    plt.title('위치-시간 그래프 (Position - Time)')
    plt.xlabel('시간 (s)')
    plt.ylabel('높이 (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('position_time.png', dpi=150)
    plt.close()
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, a_hard, label='완충재 없음 (단단한 바닥)', color='red')
    plt.plot(t, a_soft, label='완충재 있음 (충격 흡수)', color='blue', linestyle='--')
    plt.title('가속도-시간 그래프 (Acceleration - Time)')
    plt.xlabel('시간 (s)')
    plt.ylabel('가속도 (m/s²)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accel_time.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
