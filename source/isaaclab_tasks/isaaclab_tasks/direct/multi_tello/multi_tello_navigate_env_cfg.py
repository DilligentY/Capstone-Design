# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG
from isaaclab_assets import TELLOAPPROX_CFG


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
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

@configclass
class MultiTelloNavigateEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    possible_agents = ["leader", "left", "right"]
    action_spaces = {"leader" : 4, "left" : 4, "right" : 4}
    observation_spaces = {"leader" : 21, "left" : 25, "right" : 25}
    state_space = 63

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

    # Robot
    leader_robot_cfg  : ArticulationCfg = TELLOAPPROX_CFG.replace(prim_path="/World/envs/env_.*/Leader")
    
    left_robot_cfg    : ArticulationCfg = TELLOAPPROX_CFG.replace(prim_path="/World/envs/env_.*/Follower_left").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.5, -0.5, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    right_robot_cfg   : ArticulationCfg = TELLOAPPROX_CFG.replace(prim_path="/World/envs/env_.*/Follower_right").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.5, 0.5, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    # Camera Sensor
    # leader_camera = CameraCfg(
    #     prim_path=f"{leader_robot_cfg.prim_path}/leader_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

    # left_camera = CameraCfg(
    #     prim_path=f"{left_robot_cfg.prim_path}/left_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

    # right_camera = CameraCfg(
    #     prim_path=f"{right_robot_cfg.prim_path}/right_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )
    
    
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.SphereCfg(
            radius=0.3,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.54), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    
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
    # reward-related parameters
    distance_threshold = 1.5
    # Max Action Scale
    max_lin_vel_x = 1.0
    max_lin_vel_y = 1.0
    max_lin_vel_z = 0.5
    max_ang_vel_z = 0.5


    # Action Noise Model for Domain Randomization
    action_noise_model = {
        "leader" : NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="abs"),
        ),

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
        "leader" : NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),
        ),

        "left" : NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),
        ),
        
        "right" : NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),
        )
    }

