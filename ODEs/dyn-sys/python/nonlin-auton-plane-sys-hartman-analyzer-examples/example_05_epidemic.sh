python ../nonlin-auton-plane-sys-hartman-analyzer.py \
    --dx_dt "-0.003 * x * y" \
    --dy_dt "0.003 * x * y - 0.5 * y" \
    --t_num_of_samples 500 \
    --x0_begin 0 --x0_end 1000 \
    --y0_begin 0 --y0_end 1000
