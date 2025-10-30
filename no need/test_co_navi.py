#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def trilateration2D_linear(anchors, distances, z_me=1.0):
    """
    anchors: list of (x,y)
    distances: list of measured distances
    """
    anchors = np.array(anchors, dtype=float)
    distances = np.array(distances, dtype=float)

    # 选第一个锚点作为基准
    x0, y0 = anchors[0]
    d0 = distances[0]

    A = []
    b = []
    for (xi, yi), di in zip(anchors[1:], distances[1:]):
        A.append([2*(xi - x0), 2*(yi - y0)])
        b.append((d0**2 - di**2) - (x0**2 - xi**2) - (y0**2 - yi**2))

    A = np.array(A)
    b = np.array(b)

    # 最小二乘解
    xy, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return np.array([xy[0], xy[1], z_me])

# ========== 测试 ==========
anchors = [
    (-10, -7),
    (-15, -5),
    (-13, -10),
    (-17, 10)
]

distances = [10.20, 6.40, 7.07, 19.24]

for (ax, ay), d in zip(anchors, distances):
    d_true = np.linalg.norm(np.array([ax, ay]) - np.array([-20, -9]))  # 真实点 (-20,-9)
    print(f"Anchor=({ax:.1f},{ay:.1f}), measured={d:.2f}, true={d_true:.2f}")

est = trilateration2D_linear(anchors, distances, z_me=1.0)
print("Estimated position:", est)
