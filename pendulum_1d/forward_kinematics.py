# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import collections
import numpy as np
import matplotlib.pyplot as plt

from pendulum_1d import Pendulum1D


class FKPendulum(Pendulum1D):
    def init_coppelia(self):
        super().init_coppelia()
        self.sim.setObjectInt32Param(
            self.joint_0,
            self.sim.jointintparam_dynctrlmode,
            self.sim.jointdynctrl_position,
        )
        # error store queue
        self.error_queue = collections.deque(maxlen=2048)
        # plot object
        self.plt_objects = [None] * 2

    def control_joint(self):
        self.sim.setJointTargetPosition(self.joint_0, self.control.theta_0)

    def read_dummy(self):
        C1 = self.sim.getObjectPosition(self.dummy)
        return np.array(C1)

    def fk(self):
        theta_b = -np.pi / 2
        T_WB = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        theta_0 = self.control.theta_0
        T_B0 = np.array(
            [
                [np.cos(theta_0), -np.sin(theta_0), 0, 0],
                [0, 0, 1, 0.101000000000000],
                [-np.sin(theta_0), -np.cos(theta_0), 0, 0],
                [0, 0, 0, 1],
            ]
        )
        C1_hat = T_WB @ T_B0 @ np.array([-0.45, 0.0, 0.0, 1])
        return C1_hat[:-1]

    def run_step(self, count):
        # joint control
        self.control_joint()
        # read dummy position
        C1 = self.read_dummy()
        # calc fk
        C1_hat = self.fk()
        # display
        self.visualize(C1, C1_hat)

    def visualize(self, C1, C1_hat):
        self.error_queue.append(np.linalg.norm(C1 - C1_hat))
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
