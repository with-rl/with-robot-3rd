# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from pynput.keyboard import Key

from pendulum_2d import Pendulum2D


class IKPendulum(Pendulum2D):
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
        self.C2_ref = self.fk(0, 0)
        self.control.theta_0, self.control.theta_1 = self.fsolve()

    def on_press(self, key):
        if key == Key.space:
            theta_0 = (1 - np.random.rand() * 2) * np.pi
            theta_1 = (1 - np.random.rand() * 2) * np.pi
            self.C2_ref = self.fk(theta_0, theta_1)
            self.control.theta_0, self.control.theta_1 = self.fsolve()
        else:
            super().on_press(key)

    def control_joint(self):
        self.sim.setJointTargetPosition(self.joint_0, self.control.theta_0)
        self.sim.setJointTargetPosition(self.joint_1, self.control.theta_1)

    def read_dummy(self):
        C1 = self.sim.getObjectPosition(self.dummy)
        return np.array(C1)

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

    def ik(self, thetas, params):
        (theta_0, theta_1) = thetas
        (C2_ref,) = params
        C2_hat = self.fk(theta_0, theta_1)
        return np.linalg.norm(C2_ref - C2_hat), 0

    def fsolve(self):
        thetas = fsolve(
            func=self.ik,
            x0=[self.control.theta_0, self.control.theta_1],
            args=[self.C2_ref],
        )
        theta_0, theta_1 = thetas
        while np.pi < theta_0:
            theta_0 -= 2 * np.pi
        while theta_0 < -np.pi:
            theta_0 += 2 * np.pi
        while np.pi < theta_1:
            theta_1 -= 2 * np.pi
        while theta_1 < -np.pi:
            theta_1 += 2 * np.pi
        return theta_0, theta_1

    def run_step(self, count):
        # joint control
        self.control_joint()
        # read dummy position
        C2 = self.read_dummy()
        # display
        self.visualize(C2, self.C2_ref)

    def visualize(self, C2, C2_ref):
        self.error_queue.append(np.linalg.norm(C2 - C2_ref))
        # clear object
        for object in self.plt_objects:
            if object:
                object.remove()
        (self.plt_objects[0],) = plt.plot(self.error_queue, "-r", label="error")
        self.plt_objects[1] = plt.legend()
        plt.pause(0.001)


if __name__ == "__main__":
    client = IKPendulum()
    client.init_coppelia()
    client.run_coppelia()
