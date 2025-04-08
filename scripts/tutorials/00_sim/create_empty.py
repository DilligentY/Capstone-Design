# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext
from isaacsim.robot_setup.assembler import RobotAssembler,AssembledBodies 
from isaacsim.core.prims import SingleArticulation
import numpy as np


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    robot_assembler = RobotAssembler()
    base_robot_path = "/World/cube"
    attach_robot_path = "/World/cf2x_modified"
    base_robot_mount_frame = "/assembler_mount_frame"
    attach_robot_mount_frame = "/body"
    fixed_joint_offset = np.array([0.0,-0.46,0.0])
    fixed_joint_orient = np.array([1.0,0.0,0.0,0.0])
    assembled_bodies = robot_assembler.assemble_rigid_bodies(
        base_robot_path,
        attach_robot_path,
        base_robot_mount_frame,
        attach_robot_mount_frame,
        fixed_joint_offset,
        fixed_joint_orient,
        mask_all_collisions = True
)

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
