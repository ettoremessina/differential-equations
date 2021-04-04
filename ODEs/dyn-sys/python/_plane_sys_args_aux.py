import argparse

def add_xyt_params(parser):
    parser.add_argument('--t_end',
                        type=float,
                        dest='t_end',
                        default=10.,
                        required=False,
                        help='In the phase portait diagram, it is the final value of the interval of variable t (starting value of t is 0).\nFor backward time trajectories, t goes from -t_end to 0;\nfor forward time trajectories, t goes from 0 to t_end.')

    parser.add_argument('--t_num_of_samples',
                        type=int,
                        dest='t_num_of_samples',
                        default=100,
                        required=False,
                        help='In the phase portait diagram, it is the number of samples of variable t between -t_end and 0 for backward time trajectories\nand also it is the number of samples of variable t between 0 and t_end for forward time trajectories')

    parser.add_argument('--x0_begin',
                        type=float,
                        dest='x0_begin',
                        default=-5.,
                        required=False,
                        help='In the phase portait diagram, it is the starting value of the interval of initial condition x0')

    parser.add_argument('--x0_end',
                        type=float,
                        dest='x0_end',
                        default=5.,
                        required=False,
                        help='In the phase portait diagram, it is the final value of the interval of initial condition x0')

    parser.add_argument('--x0_num_of_samples',
                        type=int,
                        dest='x0_num_of_samples',
                        default=6,
                        required=False,
                        help='In the phase portait diagram, it is the number of samples of initial condition x0 between x0_begin and x0_end')

    parser.add_argument('--y0_begin',
                        type=float,
                        dest='y0_begin',
                        default=-5.,
                        required=False,
                        help='In the phase portait diagram, it is the starting value of interval for initial condition y0')

    parser.add_argument('--y0_end',
                        type=float,
                        dest='y0_end',
                        default=5.,
                        required=False,
                        help='In the phase portait diagram, it is the final value of interval for initial condition y0')

    parser.add_argument('--y0_num_of_samples',
                        type=int,
                        dest='y0_num_of_samples',
                        default=6,
                        required=False,
                        help='In the phase portait diagram, it is the number of samples of initial condition y0 between y0_begin and y0_end')

def add_plot_params(parser):
    parser.add_argument('--font_size',
                        type=int,
                        dest='font_size',
                        default=10,
                        required=False,
                        help='font size')
