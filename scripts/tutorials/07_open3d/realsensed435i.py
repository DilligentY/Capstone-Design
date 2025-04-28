import pyrealsense2 as rs
import open3d as o3d
from realsenseprocess import *


class RealSenseD435IConfig():
    def __init__(self, pipeline : rs.pipeline):
        self.fps = 30
        self.width = 640
        self.height = 480
        self.extrinsic_mat = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        self.config = rs.config()
        
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        
        profile = pipeline.start(self.config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        self.intrinsic_mat = o3d.camera.PinholeCameraIntrinsic(
                            intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)


class RealSenseD435I():
    def __init__(self):
        self.align = rs.align(rs.stream.color)
        self.pipeline = rs.pipeline()
        self.config = RealSenseD435IConfig(self.pipeline)
        self.pcd = o3d.geometry.PointCloud()
        
        
    def convert_rs_frames_to_pointcloud(self, rs_frames : rs.composite_frame) -> o3d.geometry.PointCloud:
        aligned_frames = self.align.process(rs_frames)
        rs_depth_frame = aligned_frames.get_depth_frame()
        np_depth = np.asanyarray(rs_depth_frame.get_data())
        o3d_depth = o3d.geometry.Image(np_depth)

        rs_color_frame = aligned_frames.get_color_frame()
        np_color = np.asanyarray(rs_color_frame.get_data())
        o3d_color = o3d.geometry.Image(np_color)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=4000.0, convert_rgb_to_intensity=False)

        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, self.config.intrinsic_mat, self.config.extrinsic_mat)
    
        return self.pcd




if __name__ == "__main__":
    cam = RealSenseD435I()
    
    rs_frames = cam.pipeline.wait_for_frames()
    pcd = cam.convert_rs_frames_to_pointcloud(rs_frames)
    pcd_down = Voxel_Downsampling(pcd)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualizer", 
                      width = cam.config.width, height= cam.config.height)
    
    vis.add_geometry(pcd_down)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2
    
    while True:
        rs_frames = cam.pipeline.wait_for_frames()
        pcd = cam.convert_rs_frames_to_pointcloud(rs_frames)
        pcd_down_new = Voxel_Downsampling(pcd)
        
        pcd_down.points = pcd_down_new.points
        pcd_down.colors = pcd_down_new.colors
        
        vis.update_geometry(pcd_down)
        if vis.poll_events():
            vis.update_renderer()
        else:
            break

    vis.destroy_window()
    cam.pipeline.stop()
    