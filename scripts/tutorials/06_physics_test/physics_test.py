# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import CARTPOLE_CFG, TELLOPAYLOAD_CFG  # isort:skip


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    # Each group will have a robot in it
    # Articulation
    quadrotor_cfg = TELLOPAYLOAD_CFG.copy()
    quadrotor_cfg.prim_path = "/World/Origin.*/Robot"
    tello = Articulation(cfg=quadrotor_cfg)

    # return the scene information
    scene_entities = {"tello": tello}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["tello"]
    env_id = robot._ALL_INDICES
    num_env = len(env_id)
    cube_body = robot.find_bodies("cube")[0]
    left_body = robot.find_bodies("body_left")[0]
    right_body = robot.find_bodies("body_right")[0]
    _thrust_left = torch.zeros(num_env, 1, 3, device=robot.device)
    _thrust_right = torch.zeros(num_env, 1, 3, device=robot.device)
    _torque = torch.zeros(num_env, 1, 3, device=robot.device)
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 1000 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")

        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 0.0
        _thrust_left[:, :, 2] = (2.0 * (2.0 - robot.data.root_state_w[:, 2])).clamp(0.0, 10.0).view(num_env, 1)
        _thrust_right[:, :, 2] = (1.0 * (2.0 - robot.data.root_state_w[:, 2])).clamp(0.0, 10.0).view(num_env, 1)
        _torque[:, :, 1] = 0.67
        _torque[:, :, 2] = 1.05


        # print(f"Cube_att : {robot.data.body_state_w[0, cube_body, 3:7].tolist()}")
        # print(f"Left_att : {robot.data.body_state_w[0, left_body, 3:7].tolist()}")
        # print(f"Right_att : {robot.data.body_state_w[0, right_body, 3:7].tolist()}")

        # print(f"Root_vel : {robot.data.root_lin_vel_w[0, :].tolist()}")
        # print(f"Cube_vel : {robot.data.body_state_w[0, cube_body, 7:].tolist()}")
        # print(f"Left_vel : {robot.data.body_state_w[0, left_body, 7:].tolist()}")
        # print(f"Right_vel : {robot.data.body_state_w[0, right_body, 7:].tolist()}")

        # -- apply action to the robot
        # robot.set_joint_effort_target(efforts)
        robot.set_external_force_and_torque(_thrust_left, _torque, body_ids=left_body, env_ids=env_id)
        robot.set_external_force_and_torque(_thrust_right, _torque, body_ids=right_body, env_ids=env_id)
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
