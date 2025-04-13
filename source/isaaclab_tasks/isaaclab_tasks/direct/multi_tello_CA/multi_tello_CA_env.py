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
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, subtract_frame_transforms, quat_error_magnitude
from .multi_tello_CA_env_cfg import MultiTelloCollisionAvoidanceEnvCfg


class MultiTelloCollisionAvoidanceEnv(DirectMARLEnv):
    cfg: MultiTelloCollisionAvoidanceEnvCfg

    def __init__(self, cfg: MultiTelloCollisionAvoidanceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
   
        self.leader_actions  = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.left_actions    = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.right_actions   = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        self.leader_vel      = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self.left_vel        = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self.right_vel       = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)

        self.ideal_left_line    = torch.tensor((-0.5, -0.5, 0.0), dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.ideal_right_line   = torch.tensor(( 0.5,  0.5, 0.0), dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # used to compare object position
        # self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        # self.in_hand_pos[:, 2] -= 0.04
        
        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.action_scale = torch.tensor([self.cfg.max_lin_vel_x, 
                                             self.cfg.max_lin_vel_y,
                                             self.cfg.max_lin_vel_z,
                                             self.cfg.max_ang_vel_z], device=self.device).repeat((self.num_envs, 1))
        
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
        self.object = RigidObject(self.cfg.object_cfg)
        
        # add ground plane & terrain
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        
        # add articulation and sensor to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["leader"] = self.leader
        self.scene.articulations["left"] = self.left
        self.scene.articulations["right"] = self.right
        self.scene.rigid_objects["object"] = self.object
        # self.scene.sensors["leader_camera"] = self.scene["leader_camera"]
        # self.scene.sensors["left_camera"] = self.scene["left_camera"]
        # self.scene.sensors["right_camera"] = self.scene["right_camera"]

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
        self.leader_actions[:, :] = self.actions["leader"].clamp(-1.0, 1.0) * self.action_scale
        self.left_actions[:, :] = self.actions["left"].clamp(-1.0, 1.0) * self.action_scale
        self.right_actions[:, :] = self.actions["right"].clamp(-1.0, 1.0) * self.action_scale



    def _apply_action(self) -> None:
        # Assign Actions each Agent
        current_vel = torch.hstack(
            (
                self.leader.data.root_lin_vel_w,
                self.leader.data.root_ang_vel_w[:, 2].unsqueeze(-1),
                self.left.data.root_lin_vel_w,
                self.left.data.root_ang_vel_w[:, 2].unsqueeze(-1),
                self.right.data.root_lin_vel_w,
                self.right.data.root_ang_vel_w[:, 2].unsqueeze(-1),
            )
        )
        target_vel = torch.hstack(
            (
                self.leader_actions,
                self.left_actions,
                self.right_actions
            )
        )
        # Approximate Time Delay System
        approx_responses = Low_Pass_Filter(current_vel, target_vel)
        # Assign each agent
        self.leader_vel[:, :3] = approx_responses[:, :3]
        self.leader_vel[:, 5] = approx_responses[:, 3] + (self._gravity_magnitude * self.physics_dt)
        self.left_vel[:, :3] = approx_responses[:, 4:7]
        self.left_vel[:, 5] = approx_responses[:, 7] + (self._gravity_magnitude * self.physics_dt)
        self.right_vel[:, :3] = approx_responses[:, 8:11]
        self.right_vel[:, 5] = approx_responses[:, 11] + (self._gravity_magnitude * self.physics_dt)
        # print(f"Current Leader Vel : {current_vel[0, :4]}")
        # print(f"Current Left Vel : {current_vel[0, 4:8]}")
        # print(f"Current Right Vel : {current_vel[0, 8:12]}")
        # print(f"Next Leader Input : {self.leader_vel[0, :]}")
        # print(f"Next Left Input : {self.left_vel[0, :]}")
        # print(f"Next right Input : {self.right_vel[0, :]}")

        # Insert velocity value
        self.leader.write_root_velocity_to_sim(self.leader_vel)
        self.left.write_root_velocity_to_sim(self.left_vel)
        self.right.write_root_velocity_to_sim(self.right_vel)


    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Decentralized Information for each Agent
        observations = {
            'leader' : torch.cat(
                (
                    # ---- Leader (21) ----
                    # Linear Velocity (3)
                    self.leader_lin_vel_b,
                    # Angular Velocity (3)
                    self.leader_ang_vel_b,
                    # Attitude (4)
                    self.leader_rot_w,
                    # Altitude (1)
                    self.leader_altitude_w,
                    # Object Position & Rotation (7)
                    self.object_pos_w,
                    self.object_rot_w,
                    # Position Error (3)
                    self.desired_pos_b
                ),
                dim=-1
            ),

            'left' : torch.cat(
                (
                    # ---- Left Agent (28) ----
                    # Linear Velocity (3)
                    self.left_lin_vel_b,
                    # Angular Velocity (3)
                    self.left_ang_vel_b,
                    # Attitude (4)
                    self.left_rot_w,
                    # Altitude (1)
                    self.left_altitude_w,
                    # Leader to Left (7),
                    self.leader_to_left_pos,
                    self.leader_to_left_rot,
                    # Object Position & Rotation (7)
                    self.object_pos_w,
                    self.object_rot_w,
                    # Position Error (3)
                    self.desired_pos_b_left
                ),
                dim=-1
            ),

            'right' : torch.cat(
                (
                    # ---- Right Agent (28) ----
                    # Linear Velocity (3)
                    self.right_lin_vel_b,
                    # Angular Velocity (3)
                    self.right_ang_vel_b,
                    # Attitude (4)
                    self.right_rot_w,
                    # Altitude (1)
                    self.right_altitude_w,
                    # Leader to Right (7)
                    self.leader_to_right_pos,
                    self.leader_to_right_rot,
                    # Object Position & Rotation (7)
                    self.object_pos_w,
                    self.object_rot_w,
                    # Position Error (3)
                    self.desired_pos_b_right
                ),
                dim=-1
            ),
        }

        return observations


    def _get_states(self) -> torch.Tensor:
        # Centralized Information for all Agents
        states = torch.cat(
            (
                # ---- Leader Agent (14) ----
                self.leader_lin_vel_b,
                self.leader_ang_vel_b,
                self.leader_rot_w,
                self.leader_altitude_w,
                self.desired_pos_b,
                # ---- Left Agent (24) ----
                self.left_lin_vel_b, 
                self.left_ang_vel_b, 
                self.left_rot_w, 
                self.left_altitude_w,
                self.leader_to_left_pos,
                self.leader_to_left_rot,
                self.desired_pos_b_left,
                # ---- Right Agent (24) ----
                self.right_lin_vel_b,
                self.right_ang_vel_b,
                self.right_rot_w,
                self.right_altitude_w,
                self.leader_to_right_pos,
                self.leader_to_right_rot,
                self.desired_pos_b_right,
                # ---- Object (7) ----
                self.object_pos_w,
                self.object_rot_w, 
            ),
            dim=-1,
        )

        return states


    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # compute Leader Reward
        lin_vel_leader = torch.sum(torch.square(self.leader_lin_vel_b), dim=1)
        ang_vel_leader = torch.sum(torch.square(self.leader_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self.desired_pos_b, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal)
        # compute Left Reward
        lin_vel_left = torch.sum(torch.square(self.left_lin_vel_b), dim=1)
        ang_vel_left = torch.sum(torch.square(self.left_ang_vel_b), dim=1)
        distance_to_ideal_left_line = torch.linalg.norm(self.ideal_left_line - self.leader_to_left_pos, dim=1)
        distance_to_ideal_left_line_mapped = torch.exp(-distance_to_ideal_left_line)
        # attitude_to_leader_left = quat_error_magnitude(self.left_rot_w, self.leader_rot_w)
        # attitude_to_leader_left_mapped = 1 / (1 + self.cfg.attitude_to_follower_reward_scale_1 * (attitude_to_leader_left / torch.pi))
        # compute Right Reward
        lin_vel_right = torch.sum(torch.square(self.right_lin_vel_b), dim=1)
        ang_vel_right = torch.sum(torch.square(self.right_ang_vel_b), dim=1)
        distance_to_ideal_right_line = torch.linalg.norm(self.ideal_right_line - self.leader_to_right_pos, dim=1)
        distance_to_ideal_right_line_mapped = torch.exp(-distance_to_ideal_right_line)
        # attitude_to_leader_right = quat_error_magnitude(self.right_rot_w, self.leader_rot_w)
        # attitude_to_leader_right_mapped = 1 / (1 + self.cfg.attitude_to_follower_reward_scale_1 * (attitude_to_leader_right / torch.pi))

        # make Reward Dictionary
        reward = {
            "leader" : (
                self.cfg.lin_vel_reward_scale * lin_vel_leader
                + self.cfg.ang_vel_reward_scale * ang_vel_leader
                + self.cfg.distance_to_goal_reward_scale * distance_to_goal_mapped
            ).view(self.num_envs, 1),

            "left" : (
                self.cfg.lin_vel_reward_scale * lin_vel_left
                + self.cfg.ang_vel_reward_scale * ang_vel_left
                + self.cfg.distance_to_follower_reward_scale * distance_to_ideal_left_line_mapped
            ).view(self.num_envs, 1),
                

            "right" : (
                self.cfg.lin_vel_reward_scale * lin_vel_right
                + self.cfg.ang_vel_reward_scale * ang_vel_right
                + self.cfg.distance_to_follower_reward_scale * distance_to_ideal_right_line_mapped
            ).view(self.num_envs, 1)

        }

        # log reward components
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["leader_reward"] = reward["leader"].mean()
        self.extras["log"]["left_reward"] = reward["left"].mean()
        self.extras["log"]["right_reward"] = reward["right"].mean()

        return reward


    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Update Agent States
        self._compute_intermediate_values()
        # reset when out of height or collision
        out_of_height = torch.logical_or(torch.max(self.h) > 2.0, torch.min(self.h) < 0.1)
        collision = torch.min(self.d) < 0.25
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

        # reset Leader, Follower Quadrotor and add noise to position
        for robot in self.scene.articulations.values():
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids]
            pos_noise = sample_uniform(-0.05, 0.05, (len(env_ids), 3), device=self.device)
            default_root_state[:, :3] += self.cfg.reset_dof_pos_noise * pos_noise + self._terrain.env_origins[env_ids]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
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
        # reset only goal position
        self.goal_pos[env_ids, :2]  = torch.zeros_like(self.goal_pos[env_ids, :2])
        self.goal_pos[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self.goal_pos[env_ids, 2] = torch.zeros_like(self.goal_pos[env_ids, 2]).uniform_(1.0, 1.5)

        # update goal pose and markers
        self.goal_markers.visualize(self.goal_pos, self.goal_rot)

    def _compute_intermediate_values(self):
        # data for leader quadrotor's observation
        self.leader_pos_w = self.leader.data.root_state_w[:, :3]
        self.leader_rot_w = self.leader.data.root_state_w[:, 3:7]
        self.leader_lin_vel_b = self.leader.data.root_lin_vel_b
        self.leader_ang_vel_b = self.leader.data.root_ang_vel_b
        self.leader_altitude_w = self.leader.data.root_state_w[:, 2].unsqueeze(-1)
        self.desired_pos_b, _ = subtract_frame_transforms(self.leader_pos_w, self.leader_rot_w, self.goal_pos)

        # data for left quadrotor's observation
        self.left_pos_w = self.left.data.root_state_w[:, :3]
        self.left_rot_w = self.left.data.root_state_w[:, 3:7]
        self.left_lin_vel_b = self.left.data.root_lin_vel_b
        self.left_ang_vel_b = self.left.data.root_ang_vel_b
        self.left_altitude_w = self.left.data.root_state_w[:, 2].unsqueeze(-1)
        self.desired_pos_b_left, _ = subtract_frame_transforms(self.left_pos_w, self.left_rot_w, self.goal_pos)
        self.leader_to_left_pos, self.leader_to_left_rot = subtract_frame_transforms(self.leader_pos_w, self.leader_rot_w, 
                                                                                     self.left_pos_w, self.left_rot_w)
        # data for right quadrotor's observation
        self.right_pos_w = self.right.data.root_state_w[:, :3]
        self.right_rot_w = self.right.data.root_state_w[:, 3:7]
        self.right_lin_vel_b = self.right.data.root_lin_vel_b
        self.right_ang_vel_b = self.right.data.root_ang_vel_b
        self.right_altitude_w = self.right.data.root_state_w[:, 2].unsqueeze(-1)
        self.desired_pos_b_right, _ = subtract_frame_transforms(self.right_pos_w, self.right_rot_w, self.goal_pos)
        self.leader_to_right_pos, self.leader_to_right_rot = subtract_frame_transforms(self.leader_pos_w, self.leader_rot_w, 
                                                                                       self.right_pos_w, self.right_rot_w)
        # data for quadrotor's States
        self.dist_leader_left = torch.linalg.norm(self.leader_pos_w - self.left_pos_w, dim=-1)
        self.dist_leader_right = torch.linalg.norm(self.leader_pos_w - self.right_pos_w, dim=-1)
        
        # data for object
        self.object_pos_w = self.object.data.root_state_w[:, :3]
        self.object_rot_w = self.object.data.root_state_w[:, 3:7]

        # data for Done & Reward calculation
        self.d = torch.cat((self.leader_to_left_pos, self.leader_to_right_pos), dim=-1)
        self.h = torch.cat((self.leader_altitude_w, self.left_altitude_w, self.right_altitude_w), dim=-1)

    

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

torch.jit.script
def Low_Pass_Filter(current_value, target_value, omega=1/0.001, dt=0.01):
    coeff = 1/(1 + omega * dt)
    return coeff * (current_value + omega * dt * target_value)