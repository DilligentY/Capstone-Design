import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_assets import CRAZYFLIE_CFG
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, GaussianNoiseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

@configclass
class MultiDroneSecneCfg(InteractiveSceneCfg):
    # ground
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    #lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # Robots
    leader_robot  : ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Leader"
    )
    
    left_robot    : ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Follower_left",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.5, -0.5, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*": 0.0,
            },
            joint_vel={
                "m1_joint": 200.0,
                "m2_joint": -200.0,
                "m3_joint": 200.0,
                "m4_joint": -200.0,
            },
        )
    )

    right_robot   : ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Follower_right",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.5, 0.5, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*": 0.0,
            },
            joint_vel={
                "m1_joint": 200.0,
                "m2_joint": -200.0,
                "m3_joint": 200.0,
                "m4_joint": -200.0,
            },
        )
    )
    
    

def run_simulator(sim : SimulationContext, scene : InteractiveScene):
    leader = scene.articulations["leader_robot"]
    left = scene.articulations["left_robot"]
    right = scene.articulations["right_robot"]
    sim_dt = sim.get_physics_dt()
    
    for robot in scene.articulations.values():
        robot.reset()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        default_root_state = robot.data.default_root_state.clone()
        default_root_state[:, :3] +=  scene.env_origins[:]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.write_root_pose_to_sim(default_root_state[:, :7])
        robot.write_root_velocity_to_sim(default_root_state[:, 7:])
    
    while simulation_app.is_running():
        sim.step()
        scene.update(sim_dt)
        
    

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    scene_cfg = MultiDroneSecneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    for key in scene.keys():
        robot = scene[key]
        scene.articulations[key] = robot    
    
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)
    

