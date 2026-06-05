import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from core.scene_manager import SceneManager
from core.camera_model import CameraModel
from core.view_space import generate_hemisphere_views, build_extrinsic, save_view_space, load_view_space
from simulation.ray_caster import RayCaster
from simulation.octo_mapper import OctoMapMapper
from planners.random_planner import RandomPlanner

def main():
    cfg_path = Path(__file__).parent.parent / "configs" / "week1.yaml"
    with open(cfg_path,'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cam_cfg = cfg["camera"]
    view_cfg = cfg["view_space"]
    voxel_cfg = cfg["voxel"]
    sim_cfg = cfg["simulation"]

    # ========== 1. 场景初始化 ==========
    scene_mgr = SceneManager(gt_resolution=voxel_cfg["gt_resolution"])
    model_path = Path(__file__).parent.parent / "data" / "raw_models" / "horse.ply"
    scene = scene_mgr.prepare(str(model_path) if model_path.exists() else None)

    # 分开使用：含桌面的用于射线投射，仅对象的用于覆盖率
    gt_voxels_all = scene["gt_voxels"]
    gt_voxels_object = scene["gt_voxels_object"]
    predicted_size = scene["predicted_size"]
    min_z_table = scene["min_z_table"]

    # ========== 2. 初始化相机 ==========
    camera = CameraModel(
        cam_cfg["width"], cam_cfg["height"],
        cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"],
        cam_cfg["near"], cam_cfg["far"]
    )

    # ========== 3. 生成/加载 32 个上半球视点 ==========
    vs_path = Path(__file__).parent.parent / "data" / "view_space_32_hemi.json"
    if not vs_path.exists():
        views = generate_hemisphere_views(
            n=view_cfg["num_views"],
            radius=view_cfg["radius"],
            center=np.array([0, 0, 0])
        )
        save_view_space(str(vs_path), views)
        print(f"已生成上半球视点空间: {vs_path}")
    else:
        views = load_view_space(str(vs_path))
        print(f"已加载上半球视点空间: {vs_path}")

    # ========== 4. 初始化射线投射（含桌面 Mesh 遮挡） ==========
    ray_caster = RayCaster(
        mesh=scene["mesh"],
        gt_voxels=gt_voxels_all,
        min_z_table=min_z_table,
        table_size=0.4
    )

    # ========== 5. 初始化 OctoMap（覆盖率只计算对象） ==========
    mapper = OctoMapMapper(resolution=voxel_cfg["resolution"])
    mapper.set_ground_truth(gt_voxels_object)

    # ========== 6. 初始化规划器 ==========
    planner = RandomPlanner(views)

    # ========== 7. 主循环 ==========
    max_steps = sim_cfg["max_steps"]
    coverage_history = []

    for step in range(max_steps):
        view_idx = planner.plan()
        if view_idx == -1:
            print("所有视点已访问完毕")
            break

        planner.mark_visited(view_idx)
        view = views[view_idx]
        extrinsic = build_extrinsic(view["position"], view["look_at"])

        # 获取相机原点（OctoMap 需要 sensor_origin）
        T_c2w = np.linalg.inv(extrinsic)
        sensor_origin = T_c2w[:3, 3]

        # 射线投射（对含桌面的 GT）
        visible_mask = ray_caster.query_visibility(extrinsic, camera)
        observed = gt_voxels_all[visible_mask]

        # 过滤桌面：只保留对象体素进入 OctoMap
        if len(observed) > 0:
            table_mask = observed[:, 2] > min_z_table + voxel_cfg["gt_resolution"]
            observed = observed[table_mask]

        # 更新 OctoMap（自动更新 Free / Occupied）
        mapper.add_observation(observed, sensor_origin)
        coverage = mapper.compute_coverage()
        coverage_history.append(coverage)

        print(f"Step {step+1:02d} | View {view_idx:02d} | "
              f"观测 {len(observed):5d} 体素 | 覆盖率: {coverage:.3f}")

    # ========== 8. 提取 32³ grid ==========
    grid = mapper.get_occupancy_grid(np.zeros(3), predicted_size)
    print(f"\n[32³ Grid] shape={grid.shape}, "
          f"Occupied={np.sum(grid>0.65)}, Free={np.sum(grid<0.45)}, "
          f"Unknown={np.sum((grid>=0.45)&(grid<=0.65))}")

    # ========== 9. 保存结果 ==========
    out_dir = Path(__file__).parent.parent / sim_cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(coverage_history)+1), coverage_history,
             marker="o", linewidth=2, markersize=8, color="#2E86AB")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Coverage Ratio (Object Only)", fontsize=12)
    plt.title("Week 1: Random NBV + OctoMap + Hemisphere Views", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "coverage_curve.png", dpi=150, bbox_inches="tight")
    plt.show()

    np.save(out_dir / "occupancy_grid_32.npy", grid)
    print(f"32³ grid 已保存: {out_dir / 'occupancy_grid_32.npy'}")

    # ========== 10. 可视化 ==========
    vis_geometries = []

    # 对象 GT（红色，不含桌面）
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_voxels_object)
    gt_pcd.paint_uniform_color([0.8, 0.2, 0.2])
    vis_geometries.append(gt_pcd)

    # 重建（绿色，OctoMap Occupied）
    recon_pcd = mapper.get_known_pcd()
    recon_pcd.paint_uniform_color([0.2, 0.8, 0.2])
    vis_geometries.append(recon_pcd)

    # 桌面（蓝色，可选）
    table_mask = gt_voxels_all[:, 2] <= min_z_table + voxel_cfg["gt_resolution"] + 1e-6
    table_voxels = gt_voxels_all[table_mask]
    if len(table_voxels) > 0:
        table_pcd = o3d.geometry.PointCloud()
        table_pcd.points = o3d.utility.Vector3dVector(table_voxels)
        table_pcd.paint_uniform_color([0.0, 0.0, 1.0])
        vis_geometries.append(table_pcd)

    # 视锥
    for i in [0, 8, 16, 24]:
        extr = build_extrinsic(views[i]["position"], views[i]["look_at"])
        frustum = camera.get_frustum_lineset(extr, scale=0.03)
        vis_geometries.append(frustum)

    o3d.visualization.draw_geometries(
        vis_geometries,
        window_name="GT(Red)=Object, Recon(Green)=OctoMap, Table(Blue)"
    )

if __name__ == "__main__":
    main()
