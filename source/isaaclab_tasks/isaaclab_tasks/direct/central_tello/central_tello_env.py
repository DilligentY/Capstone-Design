# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, GaussianNoiseCfg

##
# Pre-defined configs
##
from isaaclab_assets import TELLOAPPROX_CFG, TELLOPAYLOAD_CFG
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

class TELLOEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: CentralTELLOEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class CentralTELLOEnvCfg(DirectRLEnvCfg):
    # env information
    episode_length_s = 100.0
    decimation = 2
    action_space = 4
    observation_space = 20
    state_space = 0
    debug_vis = True

    ui_window_class_type = TELLOEnvWindow

    # simulation configuration
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot configuration
    robot : ArticulationCfg = TELLOPAYLOAD_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*": 0.0,
            },
            joint_vel={
                "m1_joint_left": 200.0,
                "m2_joint_left": -200.0,
                "m3_joint_left": 200.0,
                "m4_joint_left": -200.0,
                "m1_joint_right": 200.0,
                "m2_joint_right": -200.0,
                "m3_joint_right": 200.0,
                "m4_joint_right": -200.0,
            },
        )
    )

    # Actions Scale Factor
    thrust_to_weight = 1.7
    torque_scale = 0.1
    # reward scales & Change Logic
    distance_threshold = 0.2
    lin_vel_reward_scale = -0.3
    ang_vel_reward_scale = -0.1
    distance_to_goal_reward_scale = 40.0
    # Max Action Scale -> Scheduling이 필요할수도 ?
    max_lin_vel_x = 1.0
    max_lin_vel_y = 1.0
    max_lin_vel_z = 0.5
    max_ang_vel_z = 0.5

    # Noise Model for Domain Randomization
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="abs"),
    )

    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )


class CentralTELLOEnv(DirectRLEnv):
    cfg: CentralTELLOEnvCfg

    def __init__(self, cfg: CentralTELLOEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._left_thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._left_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._right_thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._right_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._center_vel = torch.zeros(self.num_envs, 6, device=self.device)
        # Goal position
        self._desired_pos_w_list = None
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._cube_body_id = self._robot.find_bodies("cube")[0]
        self._left_body_id = self._robot.find_bodies("body_left")[0]
        self._right_body_id = self._robot.find_bodies("body_right")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._actions[:, 0] *= self.cfg.max_lin_vel_x
        self._actions[:, 1] *= self.cfg.max_lin_vel_y
        self._actions[:, 2] *= self.cfg.max_lin_vel_z
        self._actions[:, 3] *= self.cfg.max_ang_vel_z
        # self._left_thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        # self._left_torque[:, 0, :] = self.cfg.torque_scale * self._actions[:, 1:4]
        # self._right_thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 4] + 1.0) / 2.0
        # self._right_torque[:, 0, :] = self.cfg.torque_scale * self._actions[:, 5:]

    def _apply_action(self):
        current_vel = torch.hstack(
            (
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_w[:, 2].unsqueeze(-1)
            )
        )
        _center_vel = Low_Pass_Filter(current_vel, self._actions)
        self._center_vel[:, :3] = _center_vel[:, :3]
        self._center_vel[:, 5] = _center_vel[:, 3]
        self._robot.write_root_velocity_to_sim(self._center_vel)
        # self._robot.set_external_force_and_torque(self._left_thrust, self._left_torque, body_ids=self._left_body_id)
        # self._robot.set_external_force_and_torque(self._right_thrust, self._right_torque, body_ids=self._right_body_id)

    def _get_observations(self) -> dict:
        """
        Args
            lin_vel_b : linear velocity vector based on body frame
            ang_vel_b : angular velocity vector based on body frame 
            projected_gravity_b : gravity vector based on body frame axis
            desired_pos_b : position of desired point based on body frame

        """
        # cube_state = self._robot.data.body_state_w[:, self._cube_body_id, :]
        # left_state = self._robot.data.body_state_w[:, self._left_body_id, :]
        # right_state = self._robot.data.body_state_w[:, self._right_body_id, :]
        altitude_root = self._robot.data.root_state_w[:, 2]
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )

        obs = torch.cat(
            [
                # 속도, 자세, 고도, 포지션 에러
                self._robot.data.root_lin_vel_w, # (3)
                self._robot.data.root_ang_vel_w, # (3)
                self._robot.data.root_quat_w,    # (4)
                self._robot.data.body_lin_vel_w[:, self._left_body_id, :].squeeze(1), # (3)
                self._robot.data.body_lin_vel_w[:, self._right_body_id, :].squeeze(1), # (3)
                desired_pos_b, # (3)
                altitude_root.unsqueeze(-1), # (1)
            ],
            dim=-1
        )

        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        """
        Args
            lin_vel : Blocking Agressive Linear Motion
            ang_vel : Blocking Agressive Rotation Motion
            distance_to_goal : Guide to approach goal point
        """
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
            # print(f"reward : {reward.item()}")

        # self._set_next_target_point(distance_to_goal)
    
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        # if died:
        #     print("Truncation")

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)
        # Reset simulation
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        # Reset action
        self._actions[env_ids] = 0.0
        self._center_vel = torch.zeros(self.num_envs, 6, device=self.device)
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-1.5, 1.5)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2]  = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.0)
        self.prev_distance = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        # self._set_desired_pos_circle(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        if self._desired_pos_w_list is not None:
            self.goal_pos_visualizer.visualize(self._desired_pos_w_list.transpose(1, 2).reshape(-1, 3))
        else:
            self.goal_pos_visualizer.visualize(self._desired_pos_w)


    #################### Customized Function ########################
        
    def _set_desired_pos_circle(self, env_ids, num_points=50):
        offset = self._desired_pos_w[env_ids, :].unsqueeze(0).permute(0, 2, 1)
        angle_step = torch.linspace(0, 2*torch.pi, num_points, device=self.device)
        height_step = torch.zeros(num_points, device=self.device)
        
        helix_mask = torch.stack([
            2*torch.cos(angle_step),
            2*torch.sin(angle_step),
            height_step
        ], dim=1)
        
        helix_mask = helix_mask.unsqueeze(-1).repeat(1, 1, len(env_ids))
        helix_w = helix_mask + offset
        
        if len(env_ids) == self.num_envs:
            self._desired_pos_w_list = helix_w
            self.point_ids = torch.zeros(self.num_envs, 1, device=self.device)
        else:
            self._desired_pos_w_list[:, :, env_ids] = helix_w
            self.point_ids[env_ids] = 0
        
        self._desired_pos_w[env_ids, :] = self._desired_pos_w_list[0, :, env_ids].T


    def _set_desired_pos_helix(self, env_ids, num_points=30):
        offset = self._desired_pos_w[env_ids, :].unsqueeze(0).permute(0, 2, 1)
        circle_center = offset - torch.tensor([2.0, 0.0, 0.0], device=self.device).view(1, 3, 1)
        angle_step = torch.linspace(0, 2*torch.pi, num_points, device=self.device)
        height_step = torch.linspace(0, 5, num_points, device=self.device)
        
        helix_mask = torch.stack([
            2*torch.cos(angle_step),
            2*torch.sin(angle_step),
            height_step
        ], dim=1)
        
        helix_mask = helix_mask.unsqueeze(-1).repeat(1, 1, len(env_ids))
        helix_w = helix_mask + circle_center
        
        if len(env_ids) == self.num_envs:
            self._desired_pos_w_list = helix_w
            self.point_ids = torch.zeros(self.num_envs, 1, device=self.device)
        else:
            self._desired_pos_w_list[:, :, env_ids] = helix_w
            self.point_ids[env_ids] = 0
        
        
    def _set_next_target_point(self, distance_to_goal : torch.Tensor):
        num_points = self._desired_pos_w_list.shape[0]
        next_env_ids = torch.where(distance_to_goal < self.cfg.distance_threshold)[0].to(device=self.device)
        if len(next_env_ids) > 0:
            self.point_ids[next_env_ids] += 1
            self.point_ids[next_env_ids] %= num_points
            self._desired_pos_w[next_env_ids, :] = self._desired_pos_w_list[self.point_ids[next_env_ids].squeeze(-1).long(), :, next_env_ids]


torch.jit.script
def Low_Pass_Filter(current_value, target_value, omega=1/0.01, dt=0.01):
    coeff = 1/(1 + omega * dt)
    return coeff * (current_value + omega * dt * target_value)

torch.jit.script
def mapping_to_link():
    pass