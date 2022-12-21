import os
import json
import numpy as np
import re
import pandas as pd
from datetime import datetime
import math


def load_results_from_dir(rootdir):
    data = []
    for trial_subdir in os.listdir(rootdir):
        for subdir in os.listdir(os.path.join(rootdir, trial_subdir)):
            for fname in os.listdir(os.path.join(rootdir, trial_subdir, subdir)):
                if fname == "result.json":
                    fpath = os.path.join(rootdir, trial_subdir, subdir, fname)
                    with open(fpath, 'r') as f:
                        parsed = json.load(f)

                    if not parsed.get("success", False):
                        # ignore non-successful runs
                        continue

                    execution_time = parsed['execution_time']
                    execution_time_all = [l[-1]['t_end'] for l in parsed['event']]
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
                    anchor_count = 0
                    if scheduler != "naivescheduler":
                        anchor_count = math.floor((W_hat + 1) / sigma) + 1
                    epsilon = parsed['sched_params']['epsilon']
                    int_feas_tol = parsed['sched_params']['int_feas_tol']
                    v = parsed['sched_params']['v'][0]
                    r_charge = parsed['sched_params']['r_charge'][0]
                    r_deplete = parsed['sched_params']['r_deplete'][0]
                    B_min = parsed['sched_params']['B_min'][0]
                    B_max = parsed['sched_params']['B_max'][0]
                    B_start = parsed['sched_params']['B_start'][0]
                    trial = trial_subdir

                    utilization, frac_charged, frac_waited = get_charging_station_utilization_slowest(parsed)
                    charged_windows, waited_windows = get_drone_charg_wait_windows(parsed)

                    data.append(
                        [os.path.join(rootdir, trial_subdir, subdir),
                         scheduler,
                         execution_time,
                         execution_time_all,
                         t_solve_total,
                         t_solve_mean,
                         n_solves,
                         voxel_size,
                         N_w,
                         N_d,
                         N_s,
                         n_waypoints,
                         W_hat,
                         pi,
                         sigma,
                         anchor_count,
                         epsilon,
                         int_feas_tol,
                         v,
                         r_charge,
                         r_deplete,
                         B_min,
                         B_max,
                         B_start,
                         utilization,
                         frac_charged,
                         frac_waited,
                         charged_windows,
                         waited_windows,
                         trial
                         ])
    df = pd.DataFrame(
        data=data,
        columns=[
            'directory',
            'scheduler',
            'execution_time',
            'execution_time_all',
            't_solve_total',
            't_solve_mean',
            'n_solves',
            'voxel_size',
            'N_w',
            'N_d',
            'N_s',
            'n_waypoints',
            'W_hat',
            'pi',
            'sigma',
            'anchor_count',
            'epsilon',
            'int_feas_tol',
            'v',
            'r_charge',
            'r_deplete',
            'B_min',
            'B_max',
            'B_start',
            'utilization',
            'frac_charged',
            'frac_waited',
            'charged_windows_drones',
            'waited_windows_drones',
            'trial'])

    # convert formats
    df['voxel_size'] = df.voxel_size.astype(float) / 10
    df['trial'] = df.trial.astype(int)

    # add new columns
    df['rescheduled'] = df.pi != np.inf

    def get_timestamp(row):
        ts_dir = os.path.split(row.directory)[-1]
        return datetime.fromisoformat(ts_dir)

    df['experiment_timestamp'] = df.apply(get_timestamp, axis=1)

    return df


def get_charging_station_utilization_all(parsed):
    """
    Extract the charging station utilization across the mission log, measured for *all* drones
    """
    mission_time_cumulative = sum([l[-1]['t_end'] for l in parsed['event']])

    time_charged_all = {}
    time_waited_all = {}

    for event_list in parsed['event']:
        for ev in event_list:
            if ev['type'] == 'charged':
                station_idx = ev['node']['id']
                t_start = ev['t_start']
                t_end = ev['t_end']
                time_charged_all[station_idx] = time_charged_all.get(station_idx, 0) + t_end - t_start
            elif ev['type'] == 'waited':
                station_idx = ev['node']['id']
                t_start = ev['t_start']
                t_end = ev['t_end']
                time_waited_all[station_idx] = time_waited_all.get(station_idx, 0) + t_end - t_start

    frac_charged = sum(time_charged_all.values()) / mission_time_cumulative
    frac_waited = sum(time_waited_all.values()) / mission_time_cumulative
    utilization = frac_charged + frac_waited

    return utilization, frac_charged, frac_waited


def get_charging_station_utilization_slowest(parsed):
    """
    Extract the charging station utilization across the mission log, measured for the slowest drone only
    """
    end_times = [ev[-1]['t_end'] for ev in parsed['event']]
    slowest_uav, slowest_time = np.argmax(end_times), np.max(end_times)

    time_charged = 0
    time_waited = 0
    for ev in parsed['event'][slowest_uav]:
        if ev['type'] == 'charged':
            t_start = ev['t_start']
            t_end = ev['t_end']
            time_charged += t_end - t_start
        elif ev['type'] == 'waited':
            t_start = ev['t_start']
            t_end = ev['t_end']
            time_waited += t_end - t_start
    frac_charged = time_charged / slowest_time
    frac_waited = time_waited / slowest_time
    utilization = frac_charged + frac_waited

    return utilization, frac_charged, frac_waited


def get_drone_charg_wait_windows(parsed):
    """
    Extract the charging and waiting windows for each drone
    """
    charged_windows = []
    waited_windows = []

    for evlist in parsed['event']:
        charged_windows_d = []
        waited_windows_d = []
        cur_type = None
        cur_station = None
        window_start = None

        stop = False
        for ev in evlist:
            if ev['type'] != cur_type:
                # end of current window
                if cur_type == 'charged':
                    # end charging
                    charged_windows_d.append((window_start, ev['t_end'], cur_station))
                elif cur_type == 'waited':
                    # end waiting
                    waited_windows_d.append((window_start, ev['t_end'], cur_station))

                cur_type = ev['type']

                if cur_type == 'charged':
                    # start of charging
                    window_start = ev['t_start']
                    cur_station = ev['node']['id']
                elif cur_type == 'waited':
                    # start of waiting
                    window_start = ev['t_end']
                    cur_station = ev['node']['id']
        charged_windows.append(charged_windows_d)
        waited_windows.append(waited_windows_d)
    return charged_windows, waited_windows
