# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG
from isaaclab_assets import TELLOAPPROX_CFG, TELLOPAYLOAD_CFG


import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, GaussianNoiseCfg
from isaaclab.utils import configclass


# @configclass
# class EventCfg:
#     """Configuration for randomization."""

#     # -- robot
#     robot_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="reset",
#         min_step_count_between_reset=720,
#         params={
#             "asset_cfg": SceneEntityCfg("right_hand"),
#             "static_friction_range": (0.7, 1.3),
#             "dynamic_friction_range": (1.0, 1.0),
#             "restitution_range": (1.0, 1.0),
#             "num_buckets": 250,
#         },
#     )
#     robot_joint_stiffness_and_damping = EventTerm(
#         func=mdp.randomize_actuator_gains,
#         min_step_count_between_reset=720,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
#             "stiffness_distribution_params": (0.75, 1.5),
#             "damping_distribution_params": (0.3, 3.0),
#             "operation": "scale",
#             "distribution": "log_uniform",
#         },
#     )
#     robot_joint_pos_limits = EventTerm(
#         func=mdp.randomize_joint_parameters,
#         min_step_count_between_reset=720,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
#             "lower_limit_distribution_params": (0.00, 0.01),
#             "upper_limit_distribution_params": (0.00, 0.01),
#             "operation": "add",
#             "distribution": "gaussian",
#         },
#     )
#     robot_tendon_properties = EventTerm(
#         func=mdp.randomize_fixed_tendon_parameters,
#         min_step_count_between_reset=720,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("right_hand", fixed_tendon_names=".*"),
#             "stiffness_distribution_params": (0.75, 1.5),
#             "damping_distribution_params": (0.3, 3.0),
#             "operation": "scale",
#             "distribution": "log_uniform",
#         },
#     )

#     # -- object
#     object_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         min_step_count_between_reset=720,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("object"),
#             "static_friction_range": (0.7, 1.3),
#             "dynamic_friction_range": (1.0, 1.0),
#             "restitution_range": (1.0, 1.0),
#             "num_buckets": 250,
#         },
#     )
#     object_scale_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         min_step_count_between_reset=720,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("object"),
#             "mass_distribution_params": (0.5, 1.5),
#             "operation": "scale",
#             "distribution": "uniform",
#         },
#     )

#     # -- scene
#     reset_gravity = EventTerm(
#         func=mdp.randomize_physics_scene_gravity,
#         mode="interval",
#         is_global_time=True,
#         interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
#         params={
#             "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
#             "operation": "add",
#             "distribution": "gaussian",
#         },
#     )


@configclass
class MultiTelloPayloadEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    possible_agents = ["left", "right"]
    action_spaces = {"left" : 4, "right" : 4}
    observation_spaces = {"left" : 15, "right" : 15}
    state_space = 30

    # simulation
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

    # robot
    
    left_robot_cfg    : ArticulationCfg = TELLOPAYLOAD_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -1.5, 0.5),
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
    

    # # in-hand object : 추후 Payload Task에서 추가할 것
    
    # object_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/object",
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.0335,
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(density=500.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.54), rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.0335,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 1.0)),
            ),
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=5.0, replicate_physics=True)

    # reset
    reset_position_noise = 0.1  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # robot constant parameters
    thrust_to_weight = 1.7
    torque_scale = 0.1
    # reward-related scales
    lin_vel_reward_scale = -0.5
    ang_vel_reward_scale = -0.1
    distance_to_goal_reward_scale = 20.0
    distance_to_follower_reward_scale = -10.0
    attitude_to_follower_reward_scale_1 = 9.0
    attitude_to_follower_reward_scale_2 = 5.0
    # reward-relaed parameters
    distance_threshold = 1.5


    # Action Noise Model for Domain Randomization
    action_noise_model = {
        "left" : NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="abs"),
        ),
        
        "right" : NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="abs"),
        )
    }

    # Observation Noise Model for Domain Randomization
    observation_noise_model = {
        "left" : NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),
        ),
        
        "right" : NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),
        )
    }

