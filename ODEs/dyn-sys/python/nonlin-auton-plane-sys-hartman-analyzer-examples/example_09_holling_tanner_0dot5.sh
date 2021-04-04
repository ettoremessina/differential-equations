python ../nonlin-auton-plane-sys-hartman-analyzer.py \
    --dx_dt "x * (1 - x/7) - 6*x*y/(7+7*x)" \
    --dy_dt "0.2*y * (1 - (0.5*y)/x)" \
    --t_num_of_samples 500 \
    --x0_begin 0 --x0_end 8 \
    --y0_begin 0 --y0_end 6
