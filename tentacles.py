import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import null_space, det

# Compute geometric and material parameters
def Get_Parameters(L):
    """Compute geometric and material parameters"""
    L_trial = 0.025
    N = int(np.round(L / L_trial))
    d_base = 0.02
    d_tip = 0.005
    rho_t = 1070
    rho_w = 1000
    E_dragon_skin = 0.0425e6
    E_ecoflex = 0.008e6
    b_dragon_skin = 0.1
    b_ecoflex = 0.05
    Cd = 2.0
    return N, d_base, d_tip, rho_t, rho_w, E_dragon_skin, E_ecoflex, b_dragon_skin, b_ecoflex, Cd

# Compute stiffness and damping distributions, which are (N -1) since the first are none
def Get_Stiff_Damp(E1, E2, d_base, d_tip, rho_t, L, N, b1, b2, c):
    """Compute stiffness and damping distributions"""
    L_i = L / (N + 1)
    d_nodes = np.linspace(d_base, d_tip, N + 1)
    d_mid = (d_nodes[:-1] + d_nodes[1:]) / 2
    V_i = np.pi * (d_mid / 2)**2 * L_i
    m_i = rho_t * V_i
    I_i = (1 / 12) * m_i * L_i**2

    i_trans = c * L / L_i
    i_full = int(i_trans)
    w = i_trans - i_full

    k_i = np.zeros(N)
    b_i = np.zeros(N)

    k_i[:i_full] = (3 * E1 * (np.pi * d_mid[:i_full]**4) / 64) / (4 * L_i)
    b_i[:i_full] = b1

    if i_full < N:
        k_i[i_full] = w * (3 * E1 * (np.pi * d_mid[i_full]**4) / 64) / (4 * L_i) + \
                      (1 - w) * (3 * E2 * (np.pi * d_mid[i_full]**4) / 64) / (4 * L_i)
        b_i[i_full] = w * b1 + (1 - w) * b2

    k_i[i_full + 1:] = (3 * E2 * (np.pi * d_mid[i_full + 1:]**4) / 64) / (4 * L_i)
    b_i[i_full + 1:] = b2

    return k_i, b_i, d_mid, L_i, I_i

# Computes position for the sawtooth wave
def pos(t, A, f):
    """Angular position input"""
    return A - A * (0.5 - sum((-1)**k * np.sin(2 * np.pi * (k + 1) * f * t) / (k + 1) for k in range(4)) / np.pi)

# Computes velocity for the sawtooth wave
def vel(t, A, f):
    """Angular velocity input"""
    return 2 * A * f * sum((-1)**k * np.cos(2 * np.pi * (k + 1) * f * t) for k in range(4))

# Computes acceleration for the sawtooth wave
def acc(t, A, f):
    """Angular acceleration input"""
    return 4 * np.pi * A * f**2 * sum((-1)**k * (k + 1) * np.sin(2 * np.pi * (k + 1) * f * t) for k in range(4))

# Defines the equations of motion, based on balance of moments
def equations_of_motion(t, y, params):
    """Compute system dynamics"""
    N = params['N']
    k_i = params['k_i']
    b_i = params['b_i']
    L_i = params['L_i']
    I_i = params['I_i']
    d_mid = params['d_mid']
    rho_w = params['rho_w']
    Cd = params['Cd']
    A = params['A']
    f = params['f']

    theta = y[:N]
    omega = y[N:]
    alpha = np.zeros(N)

    theta_diff = np.zeros(N)
    omega_diff = np.zeros(N)
    theta_des = pos(t, A, f)
    omega_des = vel(t, A, f)

    theta_diff[1:] = theta[1:] - theta[:-1]
    omega_diff[1:] = omega[1:] - omega[:-1]
    theta_diff[0] = theta[0] - theta_des
    omega_diff[0] = omega[0] - omega_des

    M_elastic_left = - k_i * theta_diff - b_i * omega_diff
    M_elastic_right = np.append(k_i[1:] * theta_diff[1:] + b_i[1:] * omega_diff[1:], 0)

    A_lat = np.pi * d_mid * L_i
    U_x = np.cumsum(omega * L_i * np.cos(theta))
    U_y = np.cumsum(omega * L_i * np.sin(theta))
    U_perp = U_x * np.cos(theta) + U_y * np.sin(theta)

    a = (U_perp - U_perp) / L_i
    b = (U_perp + U_perp) / 2
    M_hydro = - (1 / 12) * rho_w * Cd * A_lat * a * b * L_i**2

    alpha = (M_elastic_left + M_elastic_right + M_hydro) / I_i

    return np.hstack((omega, alpha))

# Runs the simulation for a minute
def simulation(params):
    """Run the dynamic simulation"""
    N = params['N']
    y0 = 1e-10 * np.random.randn(2 * N)
    t_span = (0, 60)
    t_eval = np.linspace(*t_span, 60000)
    return solve_ivp(equations_of_motion, t_span, y0, t_eval=t_eval, method='BDF', atol=1e-6, rtol=1e-4, max_step=0.001, args=(params,))

# Computes the final thrust based on the balance of forces (drag, added mass)
def compute_thrust(t, theta, omega, params):
    """Compute thrust force"""
    L_i = params['L_i']
    d_mid = params['d_mid']
    rho_w = params['rho_w']
    Cd = params['Cd']
    thrust = np.zeros(len(t))

    for i in range(len(t)):
        A_lat = np.pi * d_mid * L_i
        m_fluid = rho_w * A_lat * L_i
        U_perp = omega[:, i] * L_i
        a_perp = np.gradient(U_perp, L_i)

        F_drag = -0.5 * rho_w * Cd * A_lat * U_perp**2 * np.sign(U_perp)
        F_added_mass = -m_fluid * a_perp
        F_total = F_drag + F_added_mass

        thrust[i] = np.sum(F_total * np.sin(theta[:, i]))
    return thrust

# Computes forward velocity as a function of time based on integration
def compute_velocity(m, params, t, thrust):
    """Compute forward velocity"""
    d_mean = (params['d_base'] + params['d_tip']) / 2
    A_proj = params['L'] * d_mean
    rho_w = params['rho_w']
    Cd = params['Cd']
    dt = t[1] - t[0]

    velocity = np.zeros(len(t))
    for i in range(1, len(t)):
        F_net = thrust[i] - 0.5 * rho_w * Cd * A_proj * velocity[i - 1]**2 * np.sign(velocity[i - 1])
        acceleration = F_net / m
        velocity[i] = velocity[i - 1] + acceleration * dt
    return velocity

# Computes total tentacle mass as a cone, summing all the contributions
def compute_mass(rho_t, d_base, d_tip, L, N):
    """Compute total tentacle mass"""
    L_i = L / (N + 1)
    d_nodes = np.linspace(d_base, d_tip, N + 1)
    d_mid = (d_nodes[:-1] + d_nodes[1:]) / 2
    V_i = np.pi * (d_mid / 2)**2 * L_i
    return np.sum(rho_t * V_i)

# Computes mean thrust, efficiency and velocity (if plot=True plots thrust and velocity)
def NVE(L, c, f, A, plot=False):
    """Main function: returns mean thrust, efficiency and velocity. Optionally plots thrust and velocity."""
    N, d_base, d_tip, rho_t, rho_w, E1, E2, b1, b2, Cd = Get_Parameters(L)
    k_i, b_i, d_mid, L_i, I_i = Get_Stiff_Damp(E1, E2, d_base, d_tip, rho_t, L, N, b1, b2, c)

    params = {
        'k_i': k_i,
        'b_i': b_i,
        'L_i': L_i,
        'I_i': I_i,
        'd_mid': d_mid,
        'd_base': d_base,
        'd_tip': d_tip,
        'rho_w': rho_w,
        'Cd': Cd,
        'N': N,
        'A': A,
        'f': f,
        'L': L
    }

    sol = simulation(params)
    theta = sol.y[:N, :]
    omega = sol.y[N:, :]
    t = sol.t

    thrust = compute_thrust(t, theta, omega, params)
    mass = compute_mass(rho_t, d_base, d_tip, L, N)
    velocity = compute_velocity(mass, params, t, thrust)

    mean_thrust = np.mean(thrust)
    mean_velocity = np.mean(velocity)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax1.plot(t[:int(len(t)/3)], velocity[:int(len(t)/3)] * 100, color='#E40C2B', linewidth=2)
        ax1.axhline(np.mean(velocity) * 100, color='black', linestyle='--', linewidth=1.0)
        ax1.set_ylabel('Velocity [cm/s]')
        ax1.grid(True)

        ax2.plot(t[:int(len(t)/3)], thrust[:int(len(t)/3)] * 1000, color='#009CA6', linewidth=2)
        ax2.axhline(np.mean(thrust) * 1000, color='r', linestyle='--', linewidth=1.0)
        ax2.set_ylabel('Thrust [mN]')
        ax2.set_xlabel('Time [s]')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    return mean_thrust, mean_velocity