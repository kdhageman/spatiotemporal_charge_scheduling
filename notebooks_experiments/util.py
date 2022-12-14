import os
import json
import numpy as np
import re
import pandas as pd


def load_results_from_dir(rootdir):
    data = []
    for trial_subdir in os.listdir(rootdir):
        for subdir in os.listdir(os.path.join(rootdir, trial_subdir)):
            for fname in os.listdir(os.path.join(rootdir, trial_subdir, subdir)):
                if fname == "result.json":
                    fpath = os.path.join(rootdir, trial_subdir, subdir, fname)
                    with open(fpath, 'r') as f:
                        parsed = json.load(f)

                    execution_time = parsed['execution_time']
                    t_solve_total = sum([x['t_solve'] for x in parsed['solve_times']])
                    t_solve_mean = np.mean([x['t_solve'] for x in parsed['solve_times']])
                    n_solves = len(parsed['solve_times'])
                    m = re.match('.*vs_(\d+).*', parsed['scenario']['source_file'])
                    voxel_size = m[1] if len(m.groups()) > 0 else None
                    N_w = parsed['scenario']['nr_waypoints']
                    N_d = parsed['scenario']['nr_drones']
                    N_s = parsed['scenario']['nr_charging_stations']
                    n_waypoints = parsed['solve_times'][0]['n_remaining_waypoints']
                    scheduler = parsed['scheduler']
                    W_hat = parsed['sched_params']['W_hat']
                    pi = parsed['sched_params']['pi']
                    sigma = parsed['sched_params']['sigma']
                    epsilon = parsed['sched_params']['epsilon']
                    int_feas_tol = parsed['sched_params']['int_feas_tol']
                    v = parsed['sched_params']['v'][0]
                    r_charge = parsed['sched_params']['r_charge'][0]
                    r_deplete = parsed['sched_params']['r_deplete'][0]
                    B_min = parsed['sched_params']['B_min'][0]
                    B_max = parsed['sched_params']['B_max'][0]
                    B_start = parsed['sched_params']['B_start'][0]
                    trial = trial_subdir

                    utilization, frac_charged, frac_waited = get_charging_station_utilization(parsed)

                    data.append(
                        [os.path.join(rootdir, trial_subdir, subdir), scheduler, execution_time, t_solve_total, t_solve_mean, n_solves, voxel_size, N_w, N_d, N_s, n_waypoints, W_hat, pi, sigma, epsilon, int_feas_tol, v, r_charge, r_deplete, B_min,
                         B_max, B_start, utilization, frac_charged, frac_waited, trial])
    df = pd.DataFrame(data=data,
                      columns=['directory', 'scheduler', 'execution_time', 't_solve_total', 't_solve_mean', 'n_solves', 'voxel_size', 'N_w', 'N_d', 'N_s', 'n_waypoints', 'W_hat', 'pi', 'sigma', 'epsilon', 'int_feas_tol', 'v', 'r_charge', 'r_deplete',
                               'B_min', 'B_max', 'B_start', 'utilization', 'frac_charged', 'frac_waited', 'trial'])
    df['voxel_size'] = df.voxel_size.astype(float) / 10
    df['trial'] = df.trial.astype(int)
    df['rescheduled'] = df.pi != np.inf

    # determine experiment type
    def experiment_type(x):
        if x.scheduler == 'naivescheduler':
            return 'naive'
        if x.pi != np.inf:
            if x.W_hat == 10:
                return 'fixed_w_hat_10'
            if x.W_hat == 15:
                return 'fixed_w_hat_15'
            if x.W_hat == 40:
                return 'fixed_w_hat_40'
            else:
                raise Exception("unknown experiment..")
        if x.sigma == 3:
            return 'sigma3'
        if x.sigma == 2:
            return 'sigma2'
        return 'optimal'

    df['experiment_type'] = df.apply(experiment_type, axis=1)

    return df


def get_charging_station_utilization(parsed):
    """
    Return the utilization for each of the charging station in the parsed results
    The utilization is expressed as:
      U = (total time charged at station + total time waited at station) / execution_time
      
    @return: the utilization, the time charged and the time waited for each station
    """
    execution_time = parsed['execution_time']

    time_charged_all = {
        i: 0 for i in range(parsed['scenario']['nr_charging_stations'])
    }
    time_waited_all = {
        i: 0 for i in range(parsed['scenario']['nr_charging_stations'])
    }

    for event_list in parsed['event']:
        for ev in event_list:
            if ev['type'] == 'charged':
                station_idx = ev['node']['id']
                t_start = ev['t_start']
                t_end = ev['t_end']
                time_charged_all[station_idx] += t_end - t_start
            elif ev['type'] == 'waited':
                station_idx = ev['node']['id']
                t_start = ev['t_start']
                t_end = ev['t_end']
                time_waited_all[station_idx] += t_end - t_start
    time_charged_all = list([v / execution_time for v in time_charged_all.values()])
    time_waited_all = list([v / execution_time for v in time_waited_all.values()])

    utilization = []
    for time_charged, time_waited in zip(time_charged_all, time_waited_all):
        utilization.append(time_charged + time_waited)
    return utilization, time_charged_all, time_waited_all
