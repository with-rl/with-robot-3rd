# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import collections
import numpy as np
import matplotlib.pyplot as plt

from pendulum_2d import Pendulum2D


class FKPendulum(Pendulum2D):
    def init_coppelia(self):
        super().init_coppelia()
        self.sim.setObjectInt32Param(
            self.joint_0,
            self.sim.jointintparam_dynctrlmode,
            self.sim.jointdynctrl_position,
        )
        self.sim.setObjectInt32Param(
            self.joint_1,
            self.sim.jointintparam_dynctrlmode,
            self.sim.jointdynctrl_position,
        )
        # error store queue
        self.error_queue = collections.deque(maxlen=2048)
        # plot object
        self.plt_objects = [None] * 2

    def control_joint(self):
        self.sim.setJointTargetPosition(self.joint_0, self.control.theta_0)
        self.sim.setJointTargetPosition(self.joint_1, self.control.theta_1)

    def read_dummy(self):
        C2 = self.sim.getObjectPosition(self.dummy)
        return np.array(C2)

    def fk(self, theta_0, theta_1):
        T_SB = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        T_B0 = np.array(
            [
                [np.cos(theta_0), -np.sin(theta_0), 0, 0],
                [0, 0, 1, 0.101],
                [-np.sin(theta_0), -np.cos(theta_0), 0, 0],
                [0, 0, 0, 1],
            ]
        )
        T_01 = np.array(
            [
                [np.cos(theta_1), -np.sin(theta_1), 0, -0.4],
                [np.sin(theta_1), np.cos(theta_1), 0, 0],
                [0, 0, 1, 0.101],
                [0, 0, 0, 1],
            ]
        )
        C2_hat = T_SB @ T_B0 @ T_01 @ np.array([-0.35, 0.0, 0.0, 1])
        return C2_hat[:-1]

    def run_step(self, count):
        # joint control
        self.control_joint()
        # read dummy position
        C2 = self.read_dummy()
        # calc fk
        C2_hat = self.fk(self.control.theta_0, self.control.theta_1)
        # display
        self.visualize(C2, C2_hat)

    def visualize(self, C2, C2_hat):
        self.error_queue.append(np.linalg.norm(C2 - C2_hat))
        # clear object
        for object in self.plt_objects:
            if object:
                object.remove()
        (self.plt_objects[0],) = plt.plot(self.error_queue, "-r", label="error")
        self.plt_objects[1] = plt.legend()
        plt.pause(0.001)


if __name__ == "__main__":
    client = FKPendulum()
    client.init_coppelia()
    client.run_coppelia()
