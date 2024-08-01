import numpy as np
import matplotlib.pyplot as plt

# Initial parameters of the system
E_C = 1.0  # The capacitance of the circuit
n_g0 = 0.501  # starting n_g value (Exactly 0.5 causes inf period)
std = 0.05  # standard deviation of n_g fluctuations
T = 10000  # time range (arbitrary units)
dt = 0.01  # step size of integral, resolution of integral
num_steps = int(T / dt)  # number of steps in the integral
num_realizations = 100  # number of simulated particles to avg

P_t_all = np.zeros((num_realizations, num_steps), dtype=complex)
np.random.seed(0)  # set seed for reproducibility

# Simulate the random walk and calculate the integral
for i in range(num_realizations):  # for each realization
    n_g_fluctuations = np.random.normal(0, std, num_steps)  # create the normal distribution for the random walk
    n_g = n_g0 + n_g_fluctuations  # add the fluctuations to the starting n_g value
    delta_E = 4 * E_C * (1 - 2 * n_g)  # energy change equation
    integrated_E = np.cumsum(delta_E) * dt  # approximate the integral
    P_t_all[i] = np.exp(1j * integrated_E)  # evaluate the exponent and insert into the array

P_t_avg = np.mean(P_t_all, axis=0)  # avg the realizations of P(t)

# Plot the results
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, T, dt), np.real(P_t_avg), label="Re(P(t)) avg", color="#9e6e61")  # hotpink
plt.plot(np.arange(0, T, dt), np.imag(P_t_avg), label="Im(P(t)) avg", color="#4f3731")  # darkmagenta
plt.xlabel("t")
plt.ylabel("P(t)")
plt.title("avg Re and Im")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.arange(0, T, dt), np.abs(P_t_avg), label="|P(t)| avg", color="#9e6e61")  # lightpink
plt.xlabel("t")
plt.ylabel("|P(t)|")
plt.title("avg magnitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("output.png", transparent=True)
# plt.show()
