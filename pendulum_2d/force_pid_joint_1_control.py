# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import collections
import numpy as np
import matplotlib.pyplot as plt
from pynput.keyboard import Key

from pendulum_2d import Pendulum2D


class ForcePIDControl(Pendulum2D):
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
            self.sim.jointdynctrl_force,
        )
        # value store queue
        self.error_queue = collections.deque(maxlen=2048)
        self.force_queue = collections.deque(maxlen=2048)
        # plot object
        self.plt_objects = [None] * 6
        # p control
        self.theta_1_ref = np.pi / 4
        self.control_flag = False
        self.error_prev = 0
        self.error_sum = 0
        self.Kp = 10.5
        self.Ki = 1.5
        self.Kd = 5.5

    def on_press(self, key):
        if key == Key.space:
            self.control_flag = not self.control_flag
        else:
            super().on_press(key)

    def control_joint(self):
        self.sim.setJointTargetPosition(self.joint_0, self.control.theta_0)
        self.sim.setJointTargetForce(self.joint_1, self.control.force_1)

    def read_joint_1(self):
        theta_1 = self.sim.getJointPosition(self.joint_1)
        return theta_1

    def pid_control(self, error, error_d, error_sum):
        force_1 = self.Kp * error + self.Ki * error_sum + self.Kd * error_d
        force_1 = np.clip(force_1, -25, 25)
        return force_1

    def run_step(self, count):
        if self.control_flag:
            theta_1 = self.read_joint_1()
            error = self.theta_1_ref - theta_1
            error_d = (error - self.error_prev) * 20  # 20 steps per sec
            self.error_sum += error
            self.error_prev = error
            self.control.theta_0 = 0
            self.control.force_1 = self.pid_control(error, error_d, self.error_sum)
        else:
            self.control.theta_0 = 0
            self.control.force_1 = 0
        # joint control
        self.control_joint()

        # read joints
        theta_1 = self.read_joint_1()
        # display
        self.visualize(theta_1, self.theta_1_ref, self.control.force_1)

    def visualize(self, theta_1, theta_1_ref, force_1):
        self.error_queue.append(np.linalg.norm(theta_1 - theta_1_ref))
        self.force_queue.append(force_1)
        # clear object
        for object in self.plt_objects:
            if object:
                object.remove()
        plt.subplot(2, 1, 1)
        (self.plt_objects[0],) = plt.plot(self.error_queue, "-r", label="error")
        self.plt_objects[1] = plt.legend()
        plt.subplot(2, 1, 2)
        (self.plt_objects[2],) = plt.plot(self.force_queue, "-b", label="force 1")
        self.plt_objects[3] = plt.legend()
        plt.pause(0.001)


if __name__ == "__main__":
    client = ForcePIDControl()
    client.init_coppelia()
    client.run_coppelia()
