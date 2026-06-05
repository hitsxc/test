import numpy as np
import open3d as o3d
from pathlib import Path

class SceneManager:
    def __init__(self, gt_resolution=0.002):
        self.gt_resolution = gt_resolution

    def prepare(self, path=None):
        # 加载模型（与之前相同，省略）
        if path is not None and Path(path).exists():
            mesh = o3d.io.read_triangle_mesh(str(path))
            if len(mesh.triangles) == 0:
                pcd = o3d.io.read_point_cloud(str(path))
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = pcd.points
            else:
                pcd = mesh.sample_points_uniformly(number_of_points=200000)
            print(f"加载模型: {Path(path).name}")
        else:
            print("未提供模型，创建测试球体...")
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_uniformly(number_of_points=100000)

        points = np.asarray(pcd.points, dtype=np.float64)

        # 1. 单位自动检测
        if np.max(np.abs(points)) > 10.0:
            points *= 0.001
            print("  [单位检测] 坐标 > 10，判定为 mm，已转换为 m")

        # 2. 包围盒估计
        raw_center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - raw_center, axis=1)
        r_max = float(np.max(distances))
        l, r = 0.0, r_max
        for _ in range(20):
            mid = (l + r) / 2.0
            pct = np.mean(distances <= mid)
            if pct > 0.95:
                r = mid
            else:
                l = mid
        predicted_size = 1.2 * mid

        # 3. 尺寸标准化
        scale = 1.0
        if predicted_size > 0.1:
            scale = 0.1 / predicted_size
            points *= scale
            predicted_size = 0.1
            print(f"  [尺寸标准化] 缩放 scale={scale:.4f}，predicted_size≈0.1m")

        # 4. 平移至原点
        center = np.mean(points, axis=0)
        points -= center

        if len(mesh.vertices) > 0:
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            if np.max(np.abs(verts)) > 10.0:
                verts *= 0.001
            if scale != 1.0:
                verts *= scale
            verts -= center
            mesh.vertices = o3d.utility.Vector3dVector(verts)

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.66, 0.66, 0.66])

        # 5. 计算 min_z_table
        min_z = float(np.min(points[:, 2]))
        min_z_table = min_z - self.gt_resolution

        # 6. 【关键修改】分别构建对象体素和含桌面的体素
        gt_voxels_object = self._build_gt_voxels(points, self.gt_resolution)
        gt_voxels_all = self._add_table_to_gt(gt_voxels_object.copy(), min_z_table, self.gt_resolution)

        print(f"  [GT 体素] 对象: {len(gt_voxels_object)} 个, 含桌面: {len(gt_voxels_all)} 个, min_z_table: {min_z_table:.4f}")

        return {
            'mesh': mesh,
            'pcd': pcd,
            'points': points,
            'predicted_size': predicted_size,
            'min_z_table': min_z_table,
            'gt_voxels': gt_voxels_all,          # 含桌面，用于射线投射
            'gt_voxels_object': gt_voxels_object,  # 仅对象，用于覆盖率计算
            'scale': scale,
        }

    def _build_gt_voxels(self, points, resolution):
        indices = np.floor(points / resolution).astype(np.int32)
        unique = np.unique(indices, axis=0)
        centers = unique * resolution + resolution / 2.0
        return centers

    def _add_table_to_gt(self, gt_voxels, min_z_table, resolution):
        table_points = []
        x_range = np.arange(-0.2, 0.2 + resolution, resolution)
        y_range = np.arange(-0.2, 0.2 + resolution, resolution)
        for x in x_range:
            for y in y_range:
                table_points.append([x, y, min_z_table])
        if len(table_points) > 0:
            table_arr = np.array(table_points)
            gt_voxels = np.vstack([gt_voxels, table_arr])
        return gt_voxels
