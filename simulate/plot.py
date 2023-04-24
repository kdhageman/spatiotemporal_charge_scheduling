import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from simulate.event import Event, EventType
from simulate.node import Node, NodeType
from simulate.util import gen_colors
from util.scenario import Scenario


class SimulationAnimator:
    def __init__(self, sc: Scenario, events: Dict[int, List[Event]], schedules: Dict[int, List[Tuple[float, List[Node]]]], interval: float):
        self.logger = logging.getLogger(__name__)
        self.sc = sc
        self.events = events
        self.schedules = schedules
        self.interval = interval  # simulation interval
        self.frame_rate_interval = 50
        self.colors_drones = gen_colors(sc.N_d)

        self.n_extra_time = 4  # seconds
        self.n_extra_frames = 2 * int(np.ceil(1000 / self.frame_rate_interval))

        # percentage
        n_percs = 20
        every = int(np.ceil(self.n_frames / n_percs))
        i = every
        self.perc_times = []
        while i < len(self.frames):
            self.perc_times.append(self.frames[i])
            i += every

    @property
    def end_time(self):
        return max([l[-1].t_end for l in self.events.values()])

    @property
    def n_frames(self):
        return len(self.frames)

    @property
    def frames(self):
        res = list(np.arange(0, self.end_time, self.interval))
        if self.end_time not in res:
            res.append(self.end_time)
        res += [self.end_time] * self.n_extra_frames
        return res

    def animate(self, fname):
        fig, ax = plt.subplots()

        cur_schedules = [s[0] for s in self.schedules.values()]
        remaining_schedules = [s[1:] for s in self.schedules.values()]

        cur_events = [e[0] for e in self.events.values()]
        remaining_events = [e[1:] for e in self.events.values()]
        remaining_waypoints = []
        for d in range(self.sc.N_d):
            remaining_waypoints.append([tuple(x) for x in self.sc.positions_w[d]])

        start_pos = [np.array(l[0]) for l in self.sc.positions_w]
        start_battery = [e.pre_battery for e in cur_events]
        start_time = [0] * self.sc.N_d
        target_pos = [e.node.pos for e in cur_events]
        target_battery = [e.battery for e in cur_events]
        target_time = [e.t_end for e in cur_events]

        # set up the figure
        ax.axis("equal")
        xmin, xmax, ymin, ymax = self.sc.bounding_box()
        xmargin = 0.3 * (xmax - xmin)
        ymargin = 0.3 * (ymax - ymin)
        ax.set_xlim([xmin - xmargin, xmax + xmargin])
        ax.set_ylim([ymin - ymargin, ymax + ymargin])

        # prepare sizes of batteries
        battery_width_outer = (xmax - xmin) * 0.1
        battery_height_outer = battery_width_outer * 0.5
        battery_y_offset = battery_height_outer + 0.1
        battery_lw_outer = 1.5
        battery_padding = battery_height_outer / 3
        battery_width_inner_max = battery_width_outer - battery_padding
        battery_height_inner = battery_height_outer - battery_padding

        # plot charging stations
        x = [pos[0] for pos in self.sc.positions_S]
        y = [pos[1] for pos in self.sc.positions_S]
        plt.scatter(x, y, marker='s', s=70, c='white', edgecolor='black', zorder=-1, alpha=0.2)

        # define paths
        current_position_scatters = []
        remaining_waypoint_scatters = []
        schedule_paths = []
        schedule_waypoint_scatters = []
        schedule_charging_scatters = []
        battery_inner_patches = []
        battery_outer_patches = []
        for d in range(self.sc.N_d):
            scat = ax.scatter([], [], marker='o', s=60, color=self.colors_drones[d], zorder=10)
            current_position_scatters.append(scat)

            scat = ax.scatter([], [], marker='x', s=10, color=self.colors_drones[d], zorder=-1, alpha=0.2)
            remaining_waypoint_scatters.append(scat)

            path, = ax.plot([], [], color=self.colors_drones[d])
            schedule_paths.append(path)

            scat = ax.scatter([], [], c='white', s=40, edgecolor=self.colors_drones[d], zorder=2)
            schedule_waypoint_scatters.append(scat)

            scat = ax.scatter([], [], marker='s', s=70, c='white', edgecolor=self.colors_drones[d], zorder=2)
            schedule_charging_scatters.append(scat)

            patch = ax.add_patch(Rectangle((0, 0), battery_width_outer, battery_height_outer, color=self.colors_drones[d], linewidth=battery_lw_outer, fill=False))
            battery_outer_patches.append(patch)

            patch = ax.add_patch(Rectangle((0, 0), 0, battery_height_inner, color=self.colors_drones[d], linewidth=0, fill=True))
            battery_inner_patches.append(patch)

        def update(t):
            # renew schedule if needed
            for d in range(self.sc.N_d):
                while remaining_schedules[d] and t >= remaining_schedules[d][0]['timestamp']:
                    cur_schedules[d] = remaining_schedules[d][0]
                    remaining_schedules[d] = remaining_schedules[d][1:]

                    # trim first node(s) from schedule if drone is already there
                    while cur_schedules[d]['nodes'] and tuple(cur_schedules[d]['nodes'][0].pos) == tuple(start_pos[d]):
                        new_sched_nodes = cur_schedules[d]['nodes'][1:]
                        cur_schedules[d]['nodes'] = new_sched_nodes

            # go through all events to see if they have happened
            interpolated_pos = []
            interpolated_battery = []
            for d in range(self.sc.N_d):
                while t >= cur_events[d].t_end and remaining_events[d]:
                    ev = cur_events[d]

                    # mark seen waypoints, so it will not be plotted as a remaining waypoint
                    if ev.type == EventType.reached and ev.node.node_type == NodeType.Waypoint:
                        remaining_waypoints[d].remove(tuple(ev.node.pos))

                    # mark nodes as reached, to clean up the remaining schedule
                    if cur_schedules[d]['nodes'] and ev.type == EventType.reached and ev.node == cur_schedules[d]['nodes'][0]:
                        new_sched_nodes = cur_schedules[d]['nodes'][1:]
                        cur_schedules[d]['nodes'] = new_sched_nodes

                    # for next loop iteration
                    cur_events[d] = remaining_events[d][0]
                    remaining_events[d] = remaining_events[d][1:]

                    start_pos[d] = ev.node.pos
                    start_battery[d] = ev.battery
                    start_time[d] = ev.t_end
                    target_pos[d] = cur_events[d].node.pos
                    target_battery[d] = cur_events[d].battery
                    target_time[d] = cur_events[d].t_end

                if target_time[d] - start_time[d] == 0:
                    progress = 1
                else:
                    progress = min((t - start_time[d]) / (target_time[d] - start_time[d]), 1)
                interpolated_pos.append((1 - progress) * start_pos[d] + progress * target_pos[d])
                interpolated_battery.append((1 - progress) * start_battery[d] + progress * target_battery[d])

            # plot current position
            for d in range(self.sc.N_d):
                offsets = [interpolated_pos[d]]
                current_position_scatters[d].set_offsets(offsets)

            for d in range(self.sc.N_d):
                # outer battery
                x_outer = interpolated_pos[d][0] - (battery_width_outer / 2)
                y_outer = interpolated_pos[d][1] - (battery_height_outer / 2) - battery_y_offset
                battery_outer_patches[d].set_xy((x_outer, y_outer))

                # inner battery
                width_inner = battery_width_inner_max * interpolated_battery[d]
                x_inner = interpolated_pos[d][0] - (battery_width_inner_max / 2)
                y_inner = interpolated_pos[d][1] - (battery_height_inner / 2) - battery_y_offset
                battery_inner_patches[d].set_xy((x_inner, y_inner))
                battery_inner_patches[d].set_width(width_inner)

            # plot schedule
            for d in range(self.sc.N_d):
                # paths
                X = [interpolated_pos[d][0]] + [node.pos[0] for node in cur_schedules[d]['nodes']]
                Y = [interpolated_pos[d][1]] + [node.pos[1] for node in cur_schedules[d]['nodes']]
                schedule_paths[d].set_data(X, Y)

                # waypoints
                offsets = [(node.pos[0], node.pos[1]) for node in cur_schedules[d]['nodes'] if node.node_type == NodeType.Waypoint]
                if offsets:
                    schedule_waypoint_scatters[d].set_offsets(offsets)
                else:
                    schedule_waypoint_scatters[d].set_paths([])

                # charging stations
                offsets = [(node.pos[0], node.pos[1]) for node in cur_schedules[d]['nodes'] if node.node_type == NodeType.ChargingStation]
                if offsets:
                    schedule_charging_scatters[d].set_offsets(offsets)
                else:
                    schedule_charging_scatters[d].set_paths([])

            # plot remaining waypoints
            for d in range(self.sc.N_d):
                if remaining_waypoints[d]:
                    offsets = [(pos[0], pos[1]) for pos in remaining_waypoints[d]]
                    remaining_waypoint_scatters[d].set_offsets(offsets)
                else:
                    remaining_waypoint_scatters[d].set_paths([])

            ax.axis('off')
            ax.set_title(f"{t:.2f}s")

            # TODO: fix this!
            if t in self.perc_times:
                perc = t / self.end_time * 100
                self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] {perc:.1f}%")

            return []

        self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] animating {self.n_frames} frames")
        ani = FuncAnimation(fig, update, frames=self.frames, blit=True, interval=50)

        video = ani.to_html5_video()
        with open(fname, 'w') as f:
            f.write(video)
