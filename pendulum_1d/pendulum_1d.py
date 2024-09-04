# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
from abc import abstractmethod
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


@dataclass
class Control:
    force_0: float = 0
    theta_0: float = 0


class Pendulum1D:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.run_flag = True
        self.control = Control()

    def on_press(self, key):
        if key == Key.left:
            self.control.force_0 -= 0.1
            self.control.theta_0 -= 0.1
        if key == Key.right:
            self.control.force_0 += 0.1
            self.control.theta_0 += 0.1
        self.control.force_0 = min(max(self.control.force_0, -10), 10)

        if key == keyboard.KeyCode.from_char("q"):
            self.run_flag = False

    def init_coppelia(self):
        # Dummy
        self.dummy = self.sim.getObject("/Dummy")
        # Joints
        self.joint_0 = self.sim.getObject("/Joint_0")

    def run_coppelia(self):
        # key input
        Listener(on_press=self.on_press).start()
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()
        count = 0
        while self.run_flag:
            count += 1
            # step
            self.run_step(count)
            self.sim.step()
        self.sim.stopSimulation()

    @abstractmethod
    def run_step(self, count):
        pass
