# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import collections
import numpy as np
import matplotlib.pyplot as plt
from pynput.keyboard import Key

from pendulum_1d import Pendulum1D


class ForcePIDControl(Pendulum1D):
    def init_coppelia(self):
        super().init_coppelia()
        self.sim.setObjectInt32Param(
            self.joint_0,
            self.sim.jointintparam_dynctrlmode,
            self.sim.jointdynctrl_force,
        )
        # value store queue
        self.error_queue = collections.deque(maxlen=2048)
        self.force_queue = collections.deque(maxlen=2048)
        # plot object
        self.plt_objects = [None] * 4
        # p control
        self.theta_0_ref = 0
        self.control_flag = False
        self.error_prev = 0
        self.error_sum = 0
        self.Kp = 10.5
        self.Ki = 1.0
        self.Kd = 1.0

    def on_press(self, key):
        if key == Key.space:
            self.control_flag = not self.control_flag
        else:
            super().on_press(key)

    def control_joint(self):
        self.sim.setJointTargetForce(self.joint_0, self.control.force_0)

    def read_joint_0(self):
        theta_0 = self.sim.getJointPosition(self.joint_0)
        return theta_0

    def pid_control(self, error, error_d, error_sum):
        force_0 = self.Kp * error + self.Ki * error_sum + self.Kd * error_d
        force_0 = np.clip(force_0, -25, 25)
        return force_0

    def run_step(self, count):
        if self.control_flag:
            theta_0 = self.read_joint_0()
            error = self.theta_0_ref - theta_0
            error_d = (error - self.error_prev) * 20  # 20 steps per sec
            self.error_sum += error
            self.error_prev = error
            self.control.force_0 = self.pid_control(error, error_d, self.error_sum)
        else:
            self.control.force_0 = 0
        # joint control
        self.control_joint()
        # read joint_0
        theta_0 = self.read_joint_0()
        # display
        self.visualize(theta_0, self.theta_0_ref, self.control.force_0)

    def visualize(self, theta_0, theta_0_ref, force_0):
        self.error_queue.append(np.linalg.norm(theta_0 - theta_0_ref))
        self.force_queue.append(force_0)
        # clear object
        for object in self.plt_objects:
            if object:
                object.remove()
        plt.subplot(2, 1, 1)
        (self.plt_objects[0],) = plt.plot(self.error_queue, "-r", label="error")
        self.plt_objects[1] = plt.legend()
        plt.subplot(2, 1, 2)
        (self.plt_objects[2],) = plt.plot(self.force_queue, "-b", label="force_0")
        self.plt_objects[3] = plt.legend()
        plt.pause(0.001)


if __name__ == "__main__":
    client = ForcePIDControl()
    client.init_coppelia()
    client.run_coppelia()
