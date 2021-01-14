import numpy as np
from scipy import real, imag
from scipy.integrate import quad
from scipy.special import lambertw
import matplotlib.pyplot as plt

t_begin=0.
t_end=10.
t_nsamples=101
t_space, t_step = np.linspace(t_begin, t_end, t_nsamples, retstep=True)

k_range=9
a = 0.5
ad = -2.5
b = 1.75
h = 1.
g = lambda t : 1. - 0.1 * t
u = lambda t : 0.2 * t
x_0 = 1.5

sk_fn = lambda k :  (1./h) * lambertw(ad * h * np.exp(-a * h), k) + a
SK = [sk_fn(k) for k in range (-k_range, k_range+1)]

def x(t):
    def integrand_for_cki(t_, sk):
        return np.exp(-sk * t_) * g(t_ - h)

    def integral_for_cki(sk):
        def real_func(t_, sk):
            return np.real(integrand_for_cki(t_, sk))
        def imag_func(t_, sk):
            return np.imag(integrand_for_cki(t_, sk))
        real_integral = quad(real_func, 0., h, args=(sk))
        imag_integral = quad(imag_func, 0., h, args=(sk))
        return real_integral[0] + 1.j*imag_integral[0]

    def integrand_for_x_t(eta):
        tot = 0.
        for k in range (-k_range, k_range+1):
            sk = SK[k + k_range]
            ck_denom = (1. + ad * h * np.exp(-sk * h))
            ckn = 1. / ck_denom
            tot += np.exp(sk * (t - eta)) * ckn * b * u(eta)
        return tot

    def integral_for_x_t():
        def real_func(eta):
            return np.real(integrand_for_x_t(eta))
        def imag_func(eta):
            return np.imag(integrand_for_x_t(eta))
        real_integral = quad(real_func, 0., t)
        imag_integral = quad(imag_func, 0., t)
        return real_integral[0] + 1.j*imag_integral[0]

    tot = 0.
    for k in range (-k_range, k_range+1):
        sk = SK[k + k_range]
        int_for_cki = integral_for_cki(sk)
        ck_denom = (1. + ad * h * np.exp(-sk * h))
        cki = (x_0 + ad * np.exp(-sk * h) * int_for_cki) / ck_denom
        tot += np.exp(sk * t) * cki
    tot += integral_for_x_t()
    return tot

x_num_sol=[x(t) for t in t_space]

plt.figure()
plt.plot(t_space, np.real(x_num_sol), linewidth=1, label='real')
plt.plot(t_space, np.imag(x_num_sol), linewidth=1, label='imaginary')
plt.title('DDE 1st order IVP solved with W Lambert function')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()

num_of_cells_for_h_time = int(h/t_step)

x_num_grad_left = [np.real((x_num_sol[i+1] - x_num_sol[i])) / t_step
	for i in range(num_of_cells_for_h_time, t_nsamples-1)]
x_num_grad_right = [
	a * np.real(x_num_sol[i]) +
	ad * np.real(x_num_sol[i - num_of_cells_for_h_time])
	+ b * u(t_space[i])
		for i in range(num_of_cells_for_h_time, t_nsamples-1)]

plt.figure()
plt.plot(range(num_of_cells_for_h_time, t_nsamples-1), x_num_grad_left, linewidth=1, label='left')
plt.plot(range(num_of_cells_for_h_time, t_nsamples-1), x_num_grad_right, linewidth=1, label='right')
plt.title('Derivative of the solution of DDE solved with W Lambert function')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()
