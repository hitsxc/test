import numpy as np
import json
from pathlib import Path


def generate_hemisphere_views(n=32, radius=0.5, center=np.zeros(3)):
    """
    生成上半球均匀分布视点（对应 SCVP C++ 的 points_on_sphere + 上半球限制）
    使用面积均匀分布：z = cos(theta) 在 [0, 1] 上均匀分层
    """
    views = []
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(n):
        # 面积均匀分布：u ~ Uniform[0,1], theta = arccos(1-u)
        # 分层采样避免随机聚集
        u = (i + 0.5) / n
        theta = np.arccos(1.0 - u)  # 0 to pi/2

        phi = 2 * np.pi * i / golden_ratio

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        pos = np.array([x, y, z]) * radius + np.asarray(center)
        views.append({
            "index": i,
            "position": pos.tolist(),
            "look_at": np.asarray(center).tolist()
        })
    return views


def build_extrinsic(position, look_at, up=np.array([0, 0, 1])):
    position = np.asarray(position, dtype=float)
    look_at = np.asarray(look_at, dtype=float)
    up = np.asarray(up, dtype=float)
    z_axis = look_at - position
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-10)
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1, 0, 0])
    else:
        x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.vstack([x_axis, y_axis, z_axis]).T
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = -R.T @ position
    return extrinsic


def save_view_space(path, views):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(views, f, indent=2)


def load_view_space(path):
    with open(path, "r") as f:
        return json.load(f)
