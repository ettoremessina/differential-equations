# ODE solver demos — C++ / boost.odeint

Three demo programs that solve ODEs numerically with the
[Dormand-Prince RK45 (dopri5)](https://www.boost.org/doc/libs/release/libs/numeric/odeint/)
stepper from **boost.odeint**, using adaptive step control and dense output.
Each program compares the numerical result against the known analytical solution
and writes a CSV file with both.

---

## Programs

### `ode_1st_ord_ivp_01`

Solves the scalar 1st-order IVP:

```
dx/dt = sin(t) + 3·cos(2t) − x,   x(0) = 0,   t ∈ [0, 10]
```

Analytical solution:

```
x(t) = (1/2)·sin(t) − (1/2)·cos(t) + (3/5)·cos(2t) + (6/5)·sin(2t) − (1/10)·exp(−t)
```

Output: `ode_1st_ord_ivp_01.csv` — columns: `t`, `analytical`, `numerical`

---

### `ode_2nd_ord_ivp_01`

Solves the 2nd-order IVP reduced to a 2-D first-order system:

```
x'' + x' + 2x = 0,   x(0) = 1,   x'(0) = 0,   t ∈ [0, 12]
```

Rewritten as:

```
X[0]' =  X[1]
X[1]' = −X[1] − 2·X[0]
```

Analytical solution (x only):

```
x(t) = exp(−t/2) · ( cos(√7·t/2) + sin(√7·t/2)/√7 )
```

Output: `ode_2nd_ord_ivp_01.csv` — columns: `t`, `analytical`, `numerical`

---

### `sys_1st_ord_ivp_01`

Solves a system of 2 coupled 1st-order ODEs:

```
dx/dt = −x + y,    x(0) = 2
dy/dt =  4x − y,   y(0) = 0,   t ∈ [0, 5]
```

Analytical solutions:

```
x(t) =   exp(t) +   exp(−3t)
y(t) = 2·exp(t) − 2·exp(−3t)
```

Output: `sys_1st_ord_ivp_01.csv` — columns: `t`, `analytical_x`, `analytical_y`, `numerical_x`, `numerical_y`

---

## Prerequisites

- A C++17-capable compiler (e.g. `g++` or `clang++`)
- **[Boost](https://www.boost.org/)** with the `boost.odeint` headers installed

  On macOS with Homebrew:
  ```
  brew install boost
  ```

  On Debian/Ubuntu:
  ```
  sudo apt install libboost-dev
  ```

## Building

Before building, open [build.sh](build.sh) and set the `BOOST_INC` variable to
the directory that contains the Boost headers on your system:

```bash
BOOST_INC=/path/to/boost/include   # adjust this line
```

The default value points to a Homebrew installation
(`/opt/homebrew/Cellar/boost/1.90.0_1/include`); change it if your Boost is
installed elsewhere or has a different version.

Then run:

```
bash build.sh
```

This compiles all three programs in the current directory.

## Running

```
./ode_1st_ord_ivp_01
./ode_2nd_ord_ivp_01
./sys_1st_ord_ivp_01
```

Each program writes its CSV output to the current directory.

## Visualising the results

Open [https://csvplot.com/](https://csvplot.com/) and drag-and-drop one of the
generated CSV files onto the page. Then:

1. Drag the `t` column to the **X axis**.
2. Drag `analytical` (or `analytical_x` / `analytical_y`) to the **Y axis** to
   plot the exact solution.
3. Drag `numerical` (or `numerical_x` / `numerical_y`) to the **Y axis** to
   overlay the numerical solution and compare the two.
