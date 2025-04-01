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
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .multi_tello_navigate_cfg import MultiTelloNavigateEnvCfg


class MultiTelloNavigateEnv(DirectMARLEnv):
    cfg: MultiTelloNavigateEnvCfg

    def __init__(self, cfg: MultiTelloNavigateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.right_hand.num_joints

        # buffers for position targets
        self.right_hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.right_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.right_hand_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        
        
        self.leader_actions         = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.first_left_actions     = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.second_left_actions    = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.first_right_actions    = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.second_right_actions   = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        
        self.leader_torque          = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        self.first_left_torque      = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        self.second_left_torque     = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        self.first_right_torque     = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        self.second_right_torque    = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device)
        
        self.leader_thrust         = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        self.first_left_thrust     = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        self.second_left_thrust    = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        self.first_right_thrust    = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device) 
        self.second_right_thrust   = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device)


        # used to compare object position
        # self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        # self.in_hand_pos[:, 2] -= 0.04
        
        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([0.0, -0.64, 0.54], device=self.device)
        
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        
        # Body Id for Apply Action
        self._leader_body_id = self.leader.find_bodies("body")[0]
        self._first_left_body_id = self.first_left.find_bodies("body")[0]
        self._second_left_body_id = self.second_left.find_bodies("body")[0]
        self._first_right_body_id = self.first_right.find_bodies("body")[0]
        self._second_right_body_id = self.second_right.find_bodies("body")[0]
        
        self._robot_mass = self.leader.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.leader = Articulation(self.cfg.leader_robot_cfg)
        self.first_left = Articulation(self.cfg.first_left_robot_cfg)
        self.second_left = Articulation(self.cfg.second_left_robot_cfg)
        self.first_right = Articulation(self.cfg.first_right_robot_cfg)
        self.second_right = Articulation(self.cfg.second_right_robot_cfg)
        # self.object = RigidObject(self.cfg.object_cfg)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["leader"] = self.leader
        self.scene.articulations["first_left"] = self.first_left
        self.scene.articulations["second_left"] = self.second_left
        self.scene.articulations["first_right"] = self.first_right
        self.scene.articulations["second_right"] = self.second_right
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
        
        self.leader_actions[:, :] = self.actions["leader"].clamp(-1.0, 1.0)
        self.leader_torque[:, 0, :] = self.cfg.torque_scale * self.leader_actions[:, 1:]
        self.leader_thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.leader_actions[:, 0] + 1.0) / 2.0
        
        self.first_left_actions[:, :] = self.actions["first_left"].clamp(-1.0, 1.0)
        self.first_left_torque[:, 0, :] = self.cfg.torque_scale * self.first_left_actions[:, 1:]
        self.first_left_thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.first_left_actions[:, 0] + 1.0) / 2.0
        
        self.second_left_actions[:, :] = self.actions["second_left"].clamp(-1.0, 1.0)
        self.second_left_torque[:, 0, :] = self.cfg.torque_scale * self.leader_actions[:, 1:]
        self.second_left_thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.leader_actions[:, 0] + 1.0) / 2.0
        
        self.first_right_actions[:, :] = self.actions["first_right"].clamp(-1.0, 1.0)
        self.first_right_torque[:, 0, :] = self.cfg.torque_scale * self.leader_actions[:, 1:]
        self.first_right_thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.leader_actions[:, 0] + 1.0) / 2.0
        
        self.second_right_actions[:, :] = self.actions["second_right"].clamp(-1.0, 1.0)
        self.second_right_torque[:, 0, :] = self.cfg.torque_scale * self.leader_actions[:, 1:]
        self.second_right_thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.leader_actions[:, 0] + 1.0) / 2.0
        

    def _apply_action(self) -> None:
        # Assign Actions each Agent
        self.leader.set_external_force_and_torque(self.leader_thrust, self.leader_torque, body_ids=self._leader_body_id)
        self.first_left.set_external_force_and_torque(self.first_left_thrust, self.first_left_torque, body_ids=self._first_left_body_id)
        self.second_left.set_external_force_and_torque(self.second_left_thrust, self.second_left_torque, body_ids=self._second_left_body_id)
        self.first_right.set_external_force_and_torque(self.first_right_thrust, self.first_right_torque, body_ids=self._first_right_body_id)
        self.second_right.set_external_force_and_torque(self.second_right_thrust, self.second_right_torque, body_ids=self._second_right_body_id)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = {
            "right_hand": torch.cat(
                (
                    # ---- right hand ----
                    # DOF positions (24)
                    unscale(self.right_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                    # DOF velocities (24)
                    self.cfg.vel_obs_scale * self.right_hand_dof_vel,
                    # fingertip positions (5 * 3)
                    self.right_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                    # fingertip rotations (5 * 4)
                    self.right_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                    # fingertip linear and angular velocities (5 * 6)
                    self.right_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                    # applied actions (20)
                    self.actions["right_hand"],
                    # ---- object ----
                    # positions (3)
                    self.object_pos,
                    # rotations (4)
                    self.object_rot,
                    # linear velocities (3)
                    self.object_linvel,
                    # angular velocities (3)
                    self.cfg.vel_obs_scale * self.object_angvel,
                    # ---- goal ----
                    # positions (3)
                    self.goal_pos,
                    # rotations (4)
                    self.goal_rot,
                    # goal-object rotation diff (4)
                    quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                ),
                dim=-1,
            ),
            "left_hand": torch.cat(
                (
                    # ---- left hand ----
                    # DOF positions (24)
                    unscale(self.left_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                    # DOF velocities (24)
                    self.cfg.vel_obs_scale * self.left_hand_dof_vel,
                    # fingertip positions (5 * 3)
                    self.left_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                    # fingertip rotations (5 * 4)
                    self.left_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                    # fingertip linear and angular velocities (5 * 6)
                    self.left_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                    # applied actions (20)
                    self.actions["left_hand"],
                    # ---- object ----
                    # positions (3)
                    self.object_pos,
                    # rotations (4)
                    self.object_rot,
                    # linear velocities (3)
                    self.object_linvel,
                    # angular velocities (3)
                    self.cfg.vel_obs_scale * self.object_angvel,
                    # ---- goal ----
                    # positions (3)
                    self.goal_pos,
                    # rotations (4)
                    self.goal_rot,
                    # goal-object rotation diff (4)
                    quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                ),
                dim=-1,
            ),
        }
        return observations

    def _get_states(self) -> torch.Tensor:
        
        
        
        
        
        states = torch.cat(
            (
                # ---- right hand ----
                # DOF positions (24)
                unscale(self.right_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                # DOF velocities (24)
                self.cfg.vel_obs_scale * self.right_hand_dof_vel,
                # fingertip positions (5 * 3)
                self.right_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                # fingertip rotations (5 * 4)
                self.right_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                # fingertip linear and angular velocities (5 * 6)
                self.right_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # applied actions (20)
                self.actions["right_hand"],
                # ---- left hand ----
                # DOF positions (24)
                unscale(self.left_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                # DOF velocities (24)
                self.cfg.vel_obs_scale * self.left_hand_dof_vel,
                # fingertip positions (5 * 3)
                self.left_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                # fingertip rotations (5 * 4)
                self.left_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                # fingertip linear and angular velocities (5 * 6)
                self.left_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # applied actions (20)
                self.actions["left_hand"],
                # ---- object ----
                # positions (3)
                self.object_pos,
                # rotations (4)
                self.object_rot,
                # linear velocities (3)
                self.object_linvel,
                # angular velocities (3)
                self.cfg.vel_obs_scale * self.object_angvel,
                # ---- goal ----
                # positions (3)
                self.goal_pos,
                # rotations (4)
                self.goal_rot,
                # goal-object rotation diff (4)
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # compute reward dictionary
        goal_dist = torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)
        rew_dist = 2 * torch.exp(-self.cfg.dist_reward_scale * goal_dist)

        # log reward components
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["dist_reward"] = rew_dist.mean()
        self.extras["log"]["dist_goal"] = goal_dist.mean()

        return {"right_hand": rew_dist, "left_hand": rew_dist}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._compute_intermediate_values()

        # reset when object has fallen
        out_of_reach = self.object_pos[:, 2] <= self.cfg.fall_dist
        # reset when episode ends
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {agent: out_of_reach for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.leader._ALL_INDICES
        # reset articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

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

        # reset right hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.right_hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.right_hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.right_hand_prev_targets[env_ids] = dof_pos
        self.right_hand_curr_targets[env_ids] = dof_pos
        self.right_hand_dof_targets[env_ids] = dof_pos

        self.right_hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.right_hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # reset left hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.left_hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.left_hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.left_hand_prev_targets[env_ids] = dof_pos
        self.left_hand_curr_targets[env_ids] = dof_pos
        self.left_hand_dof_targets[env_ids] = dof_pos

        self.left_hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.left_hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

    def _compute_intermediate_values(self):
        # data for right hand
        self.right_fingertip_pos = self.right_hand.data.body_pos_w[:, self.finger_bodies]
        self.right_fingertip_rot = self.right_hand.data.body_quat_w[:, self.finger_bodies]
        self.right_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.right_fingertip_velocities = self.right_hand.data.body_vel_w[:, self.finger_bodies]

        self.right_hand_dof_pos = self.right_hand.data.joint_pos
        self.right_hand_dof_vel = self.right_hand.data.joint_vel

        # data for left hand
        self.left_fingertip_pos = self.left_hand.data.body_pos_w[:, self.finger_bodies]
        self.left_fingertip_rot = self.left_hand.data.body_quat_w[:, self.finger_bodies]
        self.left_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.left_fingertip_velocities = self.left_hand.data.body_vel_w[:, self.finger_bodies]

        self.left_hand_dof_pos = self.left_hand.data.joint_pos
        self.left_hand_dof_vel = self.left_hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w


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
