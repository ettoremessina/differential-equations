python ../nonlin-auton-plane-sys-hartman-analyzer.py \
    --dx_dt "x * (1 - x/2. - y)" \
    --dy_dt "y * (x - 1 - y/2.)" \
    --t_num_of_samples 400 \
    --x0_begin -3 --x0_end 3 \
    --y0_begin -3 --y0_end 3
