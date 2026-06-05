import numpy as np
import open3d as o3d

class CameraModel:
    def __init__(self, width, height, fx, fy, cx, cy, near=0.1, far=1.5):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.near = near
        self.far = far

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    def project(self, points_cam: np.ndarray):
        """相机坐标系 3D 点 → 图像 uv + 合法性判断"""
        z = points_cam[:, 2]
        valid = (z > self.near) & (z < self.far)

        uv = np.zeros((len(points_cam), 2))
        uv[valid, 0] = points_cam[valid, 0] / z[valid] * self.fx + self.cx
        uv[valid, 1] = points_cam[valid, 1] / z[valid] * self.fy + self.cy

        in_img = (uv[:, 0] >= 0) & (uv[:, 0] < self.width) & \
                 (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        return uv, valid & in_img

    def get_frustum_lineset(self, extrinsic, scale=0.05):
        """获取相机视锥线框（用于可视化）"""
        try:
            # Open3D 0.16+ 内置方法
            lineset = o3d.geometry.LineSet.create_camera_visualization(
                self.intrinsic, extrinsic, scale
            )
            return lineset
        except Exception:
            # 备选：手动构建视锥
            return self._build_manual_frustum(extrinsic, scale)

    def _build_manual_frustum(self, extrinsic, scale=0.05):
        """手动构建相机视锥线框"""
        near = self.near * scale
        w = near * self.width / self.fx
        h = near * self.height / self.fy

        cam_pts = np.array([
            [0, 0, 0],
            [-w / 2, -h / 2, near],
            [w / 2, -h / 2, near],
            [w / 2, h / 2, near],
            [-w / 2, h / 2, near],
        ])

        T_c2w = np.linalg.inv(extrinsic)
        cam_pts_h = np.hstack([cam_pts, np.ones((5, 1))])
        world_pts = (T_c2w @ cam_pts_h.T).T[:, :3]

        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1],
        ]

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(world_pts)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 0.5, 1.0], (len(lines), 1))
        )
        return lineset
