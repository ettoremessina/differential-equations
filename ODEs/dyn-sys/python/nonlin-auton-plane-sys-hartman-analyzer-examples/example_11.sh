python ../nonlin-auton-plane-sys-hartman-analyzer.py \
    --dx_dt "-y + x * (1 - x **2 - y ** 2)" \
    --dy_dt " x + y * (1 - x **2 - y ** 2)" \
    --t_num_of_samples 500 \
    --x0_begin -3 --x0_end 3 \
    --x0_num_of_samples 10 \
    --y0_begin -3 --y0_end 3 \
    --y0_num_of_samples 10 \
    --plot_neg_time_traj no
