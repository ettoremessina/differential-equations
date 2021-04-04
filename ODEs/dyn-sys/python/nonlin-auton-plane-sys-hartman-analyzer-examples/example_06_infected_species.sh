python ../nonlin-auton-plane-sys-hartman-analyzer.py \
    --dx_dt "(4.0 - 1.0) * x - 6.0 * y" \
    --dy_dt "1.0 * y * (x - y) - (6.0 + 1.0) * y" \
    --t_num_of_samples 500 \
    --x0_begin 0 --x0_end 20 \
    --y0_begin 0 --y0_end 20
