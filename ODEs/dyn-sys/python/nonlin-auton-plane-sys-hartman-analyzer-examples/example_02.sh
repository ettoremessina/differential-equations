python ../nonlin-auton-plane-sys-hartman-analyzer.py \
    --dx_dt "y" \
    --dy_dt "x * (1 - x**2) + y" \
    --t_num_of_samples 500 \
    --x0_begin -3 --x0_end 3 \
    --y0_begin -3 --y0_end 3
