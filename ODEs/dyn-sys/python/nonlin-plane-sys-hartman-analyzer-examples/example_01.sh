python ../nonlin-plane-sys-hartman-analyzer.py \
    --dx_dt "x" \
    --dy_dt "x**2 + y**2 - 1" \
    --t_num_of_samples 500 \
    --x0_begin -3 --x0_end 3 \
    --y0_begin -3 --y0_end 3
