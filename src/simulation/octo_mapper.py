import numpy as np
import octomap
import open3d as o3d


class OctoMapMapper:
    def __init__(self, resolution=0.00625):
        self.resolution = resolution
        self.tree = octomap.OcTree(resolution)
        self.unique_gt_voxels = None
        self.total_gt_voxels = None

    def set_ground_truth(self, gt_voxels_object: np.ndarray):
        self.gt_voxels_object = np.asarray(gt_voxels_object, dtype=np.float64)
        indices = np.floor(self.gt_voxels_object / self.resolution).astype(np.int32)
        unique_indices = np.unique(indices, axis=0)
        self.unique_gt_voxels = unique_indices * self.resolution + self.resolution / 2.0
        self.total_gt_voxels = len(self.unique_gt_voxels)
        print(f"  [OctoMap] GT 对象体素总数: {self.total_gt_voxels}（OctoMap 分辨率去重，不含桌面）")

    def add_observation(self, points: np.ndarray, sensor_origin: np.ndarray):
        if len(points) == 0:
            return
        origin = np.array([
            float(sensor_origin[0]),
            float(sensor_origin[1]),
            float(sensor_origin[2])
        ], dtype=np.float64)
        self.tree.insertPointCloud(points, origin, maxrange=-1.0)

    def _get_occupancy_safe(self, coord):
        """安全查询 occupancy，处理 NullPointerException"""
        try:
            node = self.tree.search([float(coord[0]), float(coord[1]), float(coord[2])])
            if node is not None:
                return node.getOccupancy()
        except octomap.NullPointerException:
            pass
        return 0.5  # 默认 Unknown

    def compute_coverage(self) -> float:
        if self.total_gt_voxels is None or self.total_gt_voxels == 0:
            return 0.0

        occupied = 0
        for p in self.unique_gt_voxels:
            occ = self._get_occupancy_safe(p)
            if occ > 0.65:
                occupied += 1

        return occupied / self.total_gt_voxels

    def get_occupancy_grid(self, center: np.ndarray, predicted_size: float):
        n = 32
        grid = np.zeros((n, n, n), dtype=np.float32)
        half = predicted_size
        step = 2 * half / n

        for i in range(n):
            x = center[0] - half + i * step + step / 2
            for j in range(n):
                y = center[1] - half + j * step + step / 2
                for k in range(n):
                    z = center[2] - half + k * step + step / 2
                    grid[i, j, k] = self._get_occupancy_safe([x, y, z])
        return grid

    def get_known_pcd(self):
        if self.unique_gt_voxels is None or len(self.unique_gt_voxels) == 0:
            return o3d.geometry.PointCloud()

        occupied = []
        for p in self.unique_gt_voxels:
            occ = self._get_occupancy_safe(p)
            if occ > 0.65:
                occupied.append(p)

        pcd = o3d.geometry.PointCloud()
        if len(occupied) > 0:
            pcd.points = o3d.utility.Vector3dVector(np.array(occupied))
        return pcd
