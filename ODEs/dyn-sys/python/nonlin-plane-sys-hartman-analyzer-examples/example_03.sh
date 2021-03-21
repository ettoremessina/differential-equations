python ../nonlin-plane-sys-hartman-analyzer.py \
    --dX_dt "[X[0] * (1 - X[0]/2. - X[1]), X[1] * (X[0] - 1 - X[1]/2.)]" \
    --t_num_of_samples 500 \
    --x0_begin -3 --x0_end 3 \
    --y0_begin -3 --y0_end 3
