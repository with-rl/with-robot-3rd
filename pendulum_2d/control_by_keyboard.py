# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

from pendulum_2d import Pendulum2D


class KeyboardPendulum(Pendulum2D):
    def init_coppelia(self):
        super().init_coppelia()
        self.sim.setObjectInt32Param(
            self.joint_0,
            self.sim.jointintparam_dynctrlmode,
            self.sim.jointdynctrl_force,
        )
        self.sim.setObjectInt32Param(
            self.joint_1,
            self.sim.jointintparam_dynctrlmode,
            self.sim.jointdynctrl_force,
        )

    def control_joint(self):
        self.sim.setJointTargetForce(self.joint_0, self.control.force_0)
        self.sim.setJointTargetForce(self.joint_1, self.control.force_1)

    def run_step(self, count):
        # joint control
        self.control_joint()


if __name__ == "__main__":
    client = KeyboardPendulum()
    client.init_coppelia()
    client.run_coppelia()
