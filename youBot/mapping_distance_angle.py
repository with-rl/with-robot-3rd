# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from youBot import YouBot


class Mapping(YouBot):
    def __init__(self):
        super().__init__()
        self.grid = Grid()

    def read_ref(self):
        position = self.sim.getObjectPosition(self.youBot_ref)
        orientation = self.sim.getObjectOrientation(self.youBot_ref)
        return position + orientation

    def run_step(self, count):
        # car control
        self.control_car()
        # arm control
        self.control_arm()
        # arm gripper
        self.control_gripper()
        # read position and orientation
        loc = self.read_ref()
        # update grid
        self.grid.update(loc)


@dataclass
class LidarInfo:
    offset = 0.275  # distance from youBot_ref
    alpha = 2.2  # scan max distance
    beta = np.pi / 12  # scan angle
    scan_count = 13  # scan count
    scan_theta = np.array([-np.pi / 2 + (np.pi / 12) * i for i in range(13)])
    scan_cos_min = np.cos(-np.pi / 2 - (np.pi / 12) / 2)


class Grid:
    def __init__(self):
        self.lidar_info = LidarInfo()
        # grid data
        self.grid = np.zeros((100, 100, 3))  # x, y, occupy
        self.grid[:, :, 0] = np.linspace(-4.95, 4.95, 100).reshape(1, 100)
        self.grid[:, :, 1] = np.linspace(-4.95, 4.95, 100).reshape(100, 1)
        # plot grid
        r = np.linspace(-5, 5, 101)
        p = np.linspace(-5, 5, 101)
        self.R, self.P = np.meshgrid(r, p)
        # plot object
        self.plt_objects = [None] * 3  # grid, robot, head

    def update(self, loc):
        self.mapping(loc)
        self.visualize(loc)

    def mapping(self, loc):
        x, y, z, theta_x, theta_y, theta_z = loc
        # scanner position
        rx = x + self.lidar_info.offset * np.cos(theta_z)
        ry = y + self.lidar_info.offset * np.sin(theta_z)
        scan_position = np.array([rx, ry])
        # position of grid (relative position)
        grid_xy = self.grid[:, :, :2] - scan_position.reshape(1, 1, -1)
        # distance of grid
        distance = np.linalg.norm(grid_xy, axis=-1)
        # angle of grid
        unit_xy = grid_xy / np.linalg.norm(grid_xy, axis=-1, keepdims=True)
        unit_angle = np.array([np.cos(theta_z), np.sin(theta_z)]).reshape(2, 1)
        angle = np.matmul(unit_xy, unit_angle).reshape(100, 100)
        # check valid
        valid = (distance <= self.lidar_info.alpha) * (
            angle >= self.lidar_info.scan_cos_min
        )
        self.grid[:, :, 2] = valid.astype(np.float64)

    def visualize(self, loc):
        x, y, z, theta_x, theta_y, theta_z = loc
        # clear object
        for object in self.plt_objects:
            if object:
                object.remove()
        # grid
        grid = -self.grid[:, :, 2]
        self.plt_objects[0] = plt.pcolor(self.R, self.P, grid, cmap="gray")
        # robot
        (self.plt_objects[1],) = plt.plot(
            x, y, color="green", marker="o", markersize=10
        )
        # head
        xi = x + self.lidar_info.alpha * np.cos(theta_z)
        yi = y + self.lidar_info.alpha * np.sin(theta_z)
        (self.plt_objects[2],) = plt.plot([x, xi], [y, yi], "--b")

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect("equal")
        plt.pause(0.001)


if __name__ == "__main__":
    client = Mapping()
    client.init_coppelia()
    client.run_coppelia()
