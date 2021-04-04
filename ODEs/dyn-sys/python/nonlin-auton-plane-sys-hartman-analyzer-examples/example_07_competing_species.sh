python ../nonlin-auton-plane-sys-hartman-analyzer.py \
    --dx_dt "x * (1.7 - 2.7 * x - 3.7 * y)" \
    --dy_dt "y * (1.1 - 2.1 * y - 3.1 * x)" \
    --t_num_of_samples 500 \
    --x0_begin 0 --x0_end 2 \
    --y0_begin 0 --y0_end 2
