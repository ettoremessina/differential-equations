#!/usr/bin/env bash
set -euo pipefail

BOOST_INC=/opt/homebrew/Cellar/boost/1.90.0_1/include

g++ -std=c++17 -O2 -I"$BOOST_INC" -o ode_1st_ord_ivp_01 ode_1st_ord_ivp_01.cpp
echo "Build succeeded: ode_1st_ord_ivp_01"

g++ -std=c++17 -O2 -I"$BOOST_INC" -o ode_2nd_ord_ivp_01 ode_2nd_ord_ivp_01.cpp
echo "Build succeeded: ode_2nd_ord_ivp_01"

g++ -std=c++17 -O2 -I"$BOOST_INC" -o sys_1st_ord_ivp_01 sys_1st_ord_ivp_01.cpp
echo "Build succeeded: sys_1st_ord_ivp_01"
