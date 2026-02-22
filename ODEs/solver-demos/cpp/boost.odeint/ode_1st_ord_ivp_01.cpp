// Solve the 1st-order IVP:
//
//   dx/dt = sin(t) + 3·cos(2t) - x,   x(0) = 0,   t ∈ [0, 10]
//
// Analytical solution:
//   x(t) = (1/2)·sin(t) - (1/2)·cos(t)
//         + (3/5)·cos(2t) + (6/5)·sin(2t)
//         - (1/10)·exp(-t)
//
// Numerical method: Dormand-Prince RK45 (dopri5) via boost.odeint,
// with adaptive step control and dense output — equivalent to
// SciPy's solve_ivp(..., method='RK45', dense_output=True).
//
// Output: ode_1st_ord_ivp_01.csv  (columns: t, analytical, numerical)
//
// Build:
//   g++ -std=c++17 -O2 -o ode_1st_ord_ivp_01 ode_1st_ord_ivp_01.cpp
//
// (If boost headers are not on the default search path, add e.g.:
//   -I/usr/local/include   or   -I$(brew --prefix boost)/include  )

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>

using namespace boost::numeric::odeint;

// ── types ──────────────────────────────────────────────────────────────────
using state_type = std::vector<double>;

// ── ODE right-hand side ────────────────────────────────────────────────────
void ode_fn(const state_type& x, state_type& dxdt, double t)
{
    dxdt[0] = std::sin(t) + 3.0 * std::cos(2.0 * t) - x[0];
}

// ── Analytical solution ────────────────────────────────────────────────────
double an_sol(double t)
{
    return  0.5  * std::sin(t)
          - 0.5  * std::cos(t)
          + 0.6  * std::cos(2.0 * t)
          + 1.2  * std::sin(2.0 * t)
          - 0.1  * std::exp(-t);
}

// ── main ───────────────────────────────────────────────────────────────────
int main()
{
    // Parameters matching the Python demo
    const double t_begin   = 0.0;
    const double t_end     = 10.0;
    const int    t_nsamples = 100;
    const double x_init    = 0.0;
    const double abs_err   = 1.0e-6;
    const double rel_err   = 1.0e-6;
    const double dt_init   = 0.1;          // initial adaptive step hint

    // Build the evaluation grid (linspace equivalent)
    std::vector<double> t_space(t_nsamples);
    for (int i = 0; i < t_nsamples; ++i)
        t_space[i] = t_begin + i * (t_end - t_begin) / (t_nsamples - 1);

    // Dense-output Dormand-Prince stepper (= SciPy RK45 with dense_output)
    auto stepper = make_dense_output(abs_err, rel_err,
                                     runge_kutta_dopri5<state_type>());

    // Collect numerical results at the requested time points
    std::vector<double> t_out, x_num_out;
    auto observer = [&](const state_type& x_obs, double t) {
        t_out.push_back(t);
        x_num_out.push_back(x_obs[0]);
    };

    // Integrate: the stepper advances adaptively; dense output is used
    // internally to deliver values exactly at each t in t_space.
    state_type x = {x_init};
    integrate_times(stepper, ode_fn, x,
                    t_space.begin(), t_space.end(),
                    dt_init, observer);

    // Write CSV
    const char* csv_path = "ode_1st_ord_ivp_01.csv";
    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "Cannot open " << csv_path << " for writing\n";
        return 1;
    }

    csv << "t,analytical,numerical\n";
    for (int i = 0; i < t_nsamples; ++i)
        csv << t_out[i] << "," << an_sol(t_out[i]) << "," << x_num_out[i] << "\n";

    csv.close();
    std::cout << "Solution written to " << csv_path << "\n";
    return 0;
}
