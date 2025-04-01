# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, subtract_frame_transforms, saturate

from .multi_tello_navigate_cfg import MultiTelloNavigateEnvCfg


class MultiTelloNavigateEnv(DirectMARLEnv):
    cfg: MultiTelloNavigateEnvCfg

    def __init__(self, cfg: MultiTelloNavigateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
   
        self.leader_actions   = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.left_actions     = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.right_actions    = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        
        self.leader_torque    = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        self.left_torque      = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        self.right_torque     = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        
        self.leader_thrust   = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        self.left_thrust     = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        self.right_thrust    = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 


        # used to compare object position
        # self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        # self.in_hand_pos[:, 2] -= 0.04
        
        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device).uniform_(-2.0, 2.0)
        
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        
        # Body Id for Apply Action
        self._leader_body_id = self.leader.find_bodies("body")[0]
        self._left_body_id = self.left.find_bodies("body")[0]
        self._right_body_id = self.right.find_bodies("body")[0]
        
        self._robot_mass = self.leader.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.leader = Articulation(self.cfg.leader_robot_cfg)
        self.left = Articulation(self.cfg.left_robot_cfg)
        self.right = Articulation(self.cfg.right_robot_cfg)
        # self.object = RigidObject(self.cfg.object_cfg)
        
        # add ground plane & terrain
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["leader"] = self.leader
        self.scene.articulations["left"] = self.left
        self.scene.articulations["right"] = self.right
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
        
        self.leader_actions[:, :] = self.actions["leader"].clamp(-1.0, 1.0)
        self.leader_torque[:, 0, :] = self.cfg.torque_scale * self.leader_actions[:, 1:]
        self.leader_thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.leader_actions[:, 0] + 1.0) / 2.0
        
        self.left_actions[:, :] = self.actions["left"].clamp(-1.0, 1.0)
        self.left_torque[:, 0, :] = self.cfg.torque_scale * self.left_actions[:, 1:]
        self.left_thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.left_actions[:, 0] + 1.0) / 2.0
        
        self.right_actions[:, :] = self.actions["right"].clamp(-1.0, 1.0)
        self.right_torque[:, 0, :] = self.cfg.torque_scale * self.right_actions[:, 1:]
        self.right_thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.right_actions[:, 0] + 1.0) / 2.0


    def _apply_action(self) -> None:
        # Assign Actions each Agent
        self.leader.set_external_force_and_torque(self.leader_thrust, self.leader_torque, body_ids=self._leader_body_id)
        self.left.set_external_force_and_torque(self.left_thrust, self.left_torque, body_ids=self._left_body_id)
        self.right.set_external_force_and_torque(self.right_thrust, self.right_torque, body_ids=self._right_body_id)


    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = {
            'leader' : torch.cat(
                (
                    # ---- Leader ----
                    # Linear Velocity (3)
                    self.leader_lin_vel_b,
                    # Angular Velocity (3)
                    self.leader_ang_vel_b,
                    # Attitude (4)
                    self.leader_rot,
                    # Altitude (1)
                    self.leader_altitude,
                    # Actions (4)
                    self.actions["leader"],
                    # Position Error (3)
                    self.desired_pos_b
                ),
                dim=-1
            ),

            'left' : torch.cat(
                (
                    # ---- Left Agent ----
                    # Linear Velocity (3)
                    self.left_lin_vel_b,
                    # Angular Velocity (3)
                    self.left_ang_vel_b,
                    # Attitude (4)
                    self.left_rot,
                    # Altitude (1)
                    self.left_altitude,
                    # Actions (4)
                    self.actions["left"],
                    # Position Error (3)
                    self.desired_pos_b
                ),
                dim=-1
            ),

            'right' : torch.cat(
                (
                    # ---- Right Agent ----
                    # Linear Velocity (3)
                    self.right_lin_vel_b,
                    # Angular Velocity (3)
                    self.right_ang_vel_b,
                    # Attitude (4)
                    self.right_rot,
                    # Altitude (1)
                    self.right_altitude,
                    # Actions (4)
                    self.actions["right"],
                    # Position Error (3)
                    self.desired_pos_b
                ),
                dim=-1
            ),
        }

        return observations


    def _get_states(self) -> torch.Tensor:
        states = torch.cat(
            (
                # ---- Leader Agent (18) ----
                self.leader_lin_vel_b,
                self.leader_ang_vel_b,
                self.leader_rot,
                self.leader_altitude,
                self.actions["leader"],
                self.desired_pos_b,
                # ---- Left Agent (18) ----
                self.left_lin_vel_b,
                self.left_ang_vel_b,
                self.left_rot,
                self.left_altitude,
                self.actions["left"],
                self.pos_left_leader_b,
                # ---- Right Agent (18) ----
                self.right_lin_vel_b,
                self.right_ang_vel_b,
                self.right_rot,
                self.right_altitude,
                self.actions["right"],
                self.pos_right_leader_b,
            ),
            dim=-1,
        )

        return states


    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # compute reward dictionary
        goal_dist = torch.tensor(1, dtype=torch.float, device=self.device)
        rew_dist = torch.tensor(1, dtype=torch.float, device=self.device)

        # log reward components
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["dist_reward"] = rew_dist.mean()
        self.extras["log"]["dist_goal"] = goal_dist.mean()

        return {"right_hand": rew_dist, "left_hand": rew_dist}


    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Update Agent States
        self._compute_intermediate_values()

        # reset when out of height or collision
        out_of_height = torch.logical_or(torch.max(self.h) > 2.0, torch.min(self.h) < 0.1)
        collision = torch.min(self.d) < 0.5
        truncation = torch.logical_or(out_of_height, collision)
        # reset when episode ends
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {agent: truncation for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.leader._ALL_INDICES
        # reset articulation and rigid body attributes
        super()._reset_idx(env_ids)
        self.leader.reset(env_ids)
        self.left.reset(env_ids)
        self.right.reset(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.actions["leader"][env_ids] = 0.0
        self.actions["left"][env_ids] = 0.0
        self.actions["right"][env_ids] = 0.0

        # reset goals
        self._reset_target_pose(env_ids)

        # reset Leader, Follower Quadrotor
        for robot in self.scene.articulations.values():
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, :7], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._compute_intermediate_values()

        # reset object
        # object_default_state = self.object.data.default_root_state.clone()[env_ids]
        # pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)

        # object_default_state[:, 0:3] = (
        #     object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        # )

        # rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        # object_default_state[:, 3:7] = randomize_rotation(
        #     rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )

        # object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        # self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        # self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)


    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        self.goal_pos += self._terrain.env_origins[env_ids, :2]
        self.goal_pos[env_ids, 2] = torch.zeros_like(self.goal_pos[env_ids, 2]).uniform_(0.5, 1.0)

        # update goal pose and markers
        self.goal_markers.visualize(self.goal_pos, self.goal_rot)

    def _compute_intermediate_values(self):
        # data for leader quadrotor's observation
        self.leader_pos = self.leader.data.root_state_w[:, :3]
        self.leader_rot = self.leader.data.root_state_w[:, 3:7]
        self.leader_lin_vel_b = self.leader.data.root_lin_vel_b
        self.leader_ang_vel_b = self.leader.data.root_ang_vel_b
        self.leader_altitude = self.leader.data.root_state_w[:, 2]
        self.desired_pos_b, _ = subtract_frame_transforms(self.leader_pos, self.leader_rot, self.goal_pos)

        # data for left quadrotor's observation
        self.left_pos = self.left.data.root_state_w[:, :3]
        self.left_rot = self.left.data.root_state_w[:, 3:7]
        self.left_lin_vel_b = self.left.data.root_lin_vel_b
        self.left_ang_vel_b = self.left.data.root_ang_vel_b
        self.left_altitude = self.left.data.root_state_w[:, 2]

        # data for right quadrotor's observation
        self.right_pos = self.right.data.root_state_w[:, :3]
        self.right_rot = self.right.data.root_state_w[:, 3:7]
        self.right_lin_vel_b = self.right.data.root_lin_vel_b
        self.right_ang_vel_b = self.right.data.root_ang_vel_b
        self.right_altitude = self.right.data.root_state_w[:, 2]

        # data for quadrotor's States
        self.dist_leader_left = torch.linalg.norm(self.leader_pos - self.left_pos, dim=-1)
        self.dist_leader_right = torch.linalg.norm(self.leader_pos - self.right_pos, dim=-1)
        self.dist_left_right = torch.linalg.norm(self.left_pos - self.right_pos, dim=-1)

        # data for Done & Reward calculation
        self.d = torch.cat((self.dist_leader_left, self.dist_leader_right, self.dist_left_right), dim=-1)
        self.h = torch.cat((self.leader_altitude, self.left_altitude, self.right_altitude), dim=-1)

        self.pos_left_leader_b, self.rot_left_leader_b = subtract_frame_transforms(self.leader_pos, self.leader_rot, 
                                                                                   self.left_pos, self.left_rot)
        self.pos_right_leader_b, self.rot_right_leader_b = subtract_frame_transforms(self.leader_pos, self.leader_rot,
                                                                                     self.right_pos, self.right_rot)
        

    

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
