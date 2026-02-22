// Solve the system of 2 coupled 1st-order ODEs:
//
//   dx/dt = -x + y       x(0) = 2
//   dy/dt =  4x - y      y(0) = 0
//
//   t ∈ [0, 5],  100 sample points
//
// Analytical solutions:
//   x(t) =  exp(t)  +  exp(-3t)
//   y(t) = 2·exp(t) - 2·exp(-3t)
//
// Numerical method: Dormand-Prince RK45 (dopri5) via boost.odeint,
// with adaptive step control and dense output — equivalent to
// SciPy's solve_ivp(..., method='RK45', dense_output=True).
//
// Output: sys_1st_ord_ivp_01.csv
//         columns: t, analytical_x, analytical_y, numerical_x, numerical_y
//
// Build:
//   g++ -std=c++17 -O2 \
//       -I/opt/homebrew/Cellar/boost/1.90.0_1/include \
//       -o sys_1st_ord_ivp_01 sys_1st_ord_ivp_01.cpp

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>

using namespace boost::numeric::odeint;

// ── types ──────────────────────────────────────────────────────────────────
using state_type = std::vector<double>;  // [x, y]

// ── ODE system (right-hand side) ───────────────────────────────────────────
void ode_sys(const state_type& XY, state_type& dXYdt, double /*t*/)
{
    dXYdt[0] = -XY[0] + XY[1];          // dx/dt = -x + y
    dXYdt[1] =  4.0 * XY[0] - XY[1];   // dy/dt =  4x - y
}

// ── Analytical solutions ───────────────────────────────────────────────────
double an_sol_x(double t) { return std::exp(t) + std::exp(-3.0 * t); }
double an_sol_y(double t) { return 2.0 * std::exp(t) - 2.0 * std::exp(-3.0 * t); }

// ── main ───────────────────────────────────────────────────────────────────
int main()
{
    const double t_begin    = 0.0;
    const double t_end      = 5.0;
    const int    t_nsamples = 100;
    const double x_init     = 2.0;
    const double y_init     = 0.0;
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

    // Collect both x and y at each requested time point
    std::vector<double> t_out, x_num_out, y_num_out;
    auto observer = [&](const state_type& XY, double t) {
        t_out.push_back(t);
        x_num_out.push_back(XY[0]);
        y_num_out.push_back(XY[1]);
    };

    // Integrate with dense output at the exact t_space points
    state_type XY = {x_init, y_init};
    integrate_times(stepper, ode_sys, XY,
                    t_space.begin(), t_space.end(),
                    dt_init, observer);

    // Write CSV
    const char* csv_path = "sys_1st_ord_ivp_01.csv";
    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "Cannot open " << csv_path << " for writing\n";
        return 1;
    }

    csv << "t,analytical_x,analytical_y,numerical_x,numerical_y\n";
    for (int i = 0; i < t_nsamples; ++i)
        csv << t_out[i]
            << "," << an_sol_x(t_out[i])
            << "," << an_sol_y(t_out[i])
            << "," << x_num_out[i]
            << "," << y_num_out[i]
            << "\n";

    csv.close();
    std::cout << "Solution written to " << csv_path << "\n";
    return 0;
}
