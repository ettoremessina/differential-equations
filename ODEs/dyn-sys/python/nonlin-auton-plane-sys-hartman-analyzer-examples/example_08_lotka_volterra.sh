python ../nonlin-auton-plane-sys-hartman-analyzer.py \
    --dx_dt "x * (0.666 - 1.333 * y)" \
    --dy_dt "y * (0.9 * x - 0.9)" \
    --t_num_of_samples 500 \
    --x0_begin 0 --x0_end 4 \
    --y0_begin 0 --y0_end 2
