import numpy as np
import open3d as o3d


class RayCaster:
    def __init__(self, mesh, gt_voxels, min_z_table, table_size=0.4):
        self.gt_voxels = np.asarray(gt_voxels, dtype=np.float32)
        self.min_z_table = min_z_table
        self.scene = None

        # 1. 加入对象 Mesh
        if mesh is not None and len(mesh.triangles) > 0:
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(mesh_t)
            print(f"RayCaster: Mesh 模式，GT 体素 {len(gt_voxels)} 个")
        else:
            print(f"RayCaster: 无 Mesh，GT 体素 {len(gt_voxels)} 个")

        # 2. 【新增】构建桌面 Mesh 并加入场景（与 C++ 一致）
        if self.scene is not None:
            self._add_table_to_scene(table_size)

    def _add_table_to_scene(self, table_size=0.4):
        """构建桌面平面 Mesh，加入射线投射场景"""
        half = table_size / 2
        z = self.min_z_table

        # 顶点 (4个)
        vertices = np.array([
            [-half, -half, z],
            [half, -half, z],
            [half, half, z],
            [-half, half, z],
        ], dtype=np.float32)

        # 三角形 (2个)
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)  # 注意：用 int32 而不是 uint32

        # 【修复】Open3D t.geometry.TriangleMesh 的正确构造方式
        table_mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices),  # vertex_positions
            o3d.core.Tensor(triangles)  # triangle_indices
        )
        self.scene.add_triangles(table_mesh)
        print(f"  [桌面遮挡] 已加入 {table_size}×{table_size}m 桌面到射线场景")

    def query_visibility(self, extrinsic, camera):
        """判断 gt_voxels 中哪些点可见（考虑桌面遮挡）"""
        if len(self.gt_voxels) == 0:
            return np.array([], dtype=bool)

        # 1. 转到相机坐标系
        pts_h = np.hstack([self.gt_voxels, np.ones((len(self.gt_voxels), 1))])
        pts_cam = (extrinsic @ pts_h.T).T[:, :3]

        # 2. 投影 & 视锥剔除
        uv, in_frustum = camera.project(pts_cam)
        if not np.any(in_frustum):
            return np.zeros(len(self.gt_voxels), dtype=bool)

        valid_idx = np.where(in_frustum)[0]
        valid_world = self.gt_voxels[valid_idx]
        valid_cam = pts_cam[valid_idx]

        # 3. 从相机原点发射射线
        T_c2w = np.linalg.inv(extrinsic)
        origin = (T_c2w @ np.array([0, 0, 0, 1]))[:3]
        origins = np.tile(origin, (len(valid_idx), 1))
        directions = valid_world - origins
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / (norms + 1e-10)

        rays = np.hstack([origins, directions]).astype(np.float32)

        # 4. 批量射线投射（现在桌面会遮挡！）
        if self.scene is not None:
            ans = self.scene.cast_rays(o3d.core.Tensor(rays))
            hit_dist = ans["t_hit"].numpy()
            true_dist = np.linalg.norm(valid_world - origins, axis=1)
            eps = 0.005
            visible = (hit_dist >= true_dist - eps) | np.isinf(hit_dist)
        else:
            visible = np.ones(len(valid_idx), dtype=bool)

        result = np.zeros(len(self.gt_voxels), dtype=bool)
        result[valid_idx[visible]] = True
        return result
