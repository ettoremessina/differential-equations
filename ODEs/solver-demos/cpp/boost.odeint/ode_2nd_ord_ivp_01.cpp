// Solve the 2nd-order IVP reduced to a 2D first-order system:
//
//   x'' + x' + 2x = 0
//
// Rewritten as:
//   X[0]' = X[1]             (x' = dx/dt)
//   X[1]' = -X[1] - 2*X[0]  (x'' = -x' - 2x)
//
// Initial conditions: x(0) = 1,  x'(0) = 0
// Time interval: [0, 12],  100 sample points
//
// Analytical solution (for x only):
//   x(t) = exp(-t/2) * ( cos(√7·t/2) + sin(√7·t/2)/√7 )
//
// Numerical method: Dormand-Prince RK45 (dopri5) via boost.odeint,
// with adaptive step control and dense output — equivalent to
// SciPy's solve_ivp(..., method='RK45', dense_output=True).
//
// Output: ode_2nd_ord_ivp_01.csv  (columns: t, analytical, numerical)
//
// Build:
//   g++ -std=c++17 -O2 \
//       -I/opt/homebrew/Cellar/boost/1.90.0_1/include \
//       -o ode_2nd_ord_ivp_01 ode_2nd_ord_ivp_01.cpp

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>

using namespace boost::numeric::odeint;

// ── types ──────────────────────────────────────────────────────────────────
using state_type = std::vector<double>;  // [x, dx/dt]

// ── ODE system (right-hand side) ───────────────────────────────────────────
void ode_sys(const state_type& X, state_type& dXdt, double /*t*/)
{
    dXdt[0] =  X[1];               // dx/dt   = X[1]
    dXdt[1] = -X[1] - 2.0 * X[0]; // d²x/dt² = -x' - 2x
}

// ── Analytical solution for x(t) ───────────────────────────────────────────
double an_sol_x(double t)
{
    const double s7 = std::sqrt(7.0);
    return std::exp(-t / 2.0) * (std::cos(s7 * t / 2.0) +
                                  std::sin(s7 * t / 2.0) / s7);
}

// ── main ───────────────────────────────────────────────────────────────────
int main()
{
    const double t_begin    = 0.0;
    const double t_end      = 12.0;
    const int    t_nsamples = 100;
    const double x_init     = 1.0;
    const double dxdt_init  = 0.0;
    const double abs_err    = 1.0e-6;
    const double rel_err    = 1.0e-6;
    const double dt_init    = 0.1;

    // Build evaluation grid (linspace equivalent)
    std::vector<double> t_space(t_nsamples);
    for (int i = 0; i < t_nsamples; ++i)
        t_space[i] = t_begin + i * (t_end - t_begin) / (t_nsamples - 1);

    // Dense-output Dormand-Prince stepper
    auto stepper = make_dense_output(abs_err, rel_err,
                                     runge_kutta_dopri5<state_type>());

    // Collect x (index 0) at each requested time point
    std::vector<double> t_out, x_num_out;
    auto observer = [&](const state_type& X, double t) {
        t_out.push_back(t);
        x_num_out.push_back(X[0]);   // only x, not dx/dt
    };

    // Integrate with dense output at the exact t_space points
    state_type X = {x_init, dxdt_init};
    integrate_times(stepper, ode_sys, X,
                    t_space.begin(), t_space.end(),
                    dt_init, observer);

    // Write CSV
    const char* csv_path = "ode_2nd_ord_ivp_01.csv";
    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "Cannot open " << csv_path << " for writing\n";
        return 1;
    }

    csv << "t,analytical,numerical\n";
    for (int i = 0; i < t_nsamples; ++i)
        csv << t_out[i] << "," << an_sol_x(t_out[i]) << "," << x_num_out[i] << "\n";

    csv.close();
    std::cout << "Solution written to " << csv_path << "\n";
    return 0;
}
