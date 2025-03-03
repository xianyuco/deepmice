import math

import numpy as np


def normalize_angle(x):
    x = x + math.pi
    n = x // (math.pi * 2)
    x = x - (n * 2 * math.pi + math.pi)
    return x


def quaternion_multiply(quaternion1, quaternion2):
    a, b, c, d = quaternion1
    ar, br, cr, dr = quaternion2

    at = a * ar - b * br - c * cr - d * dr

    bt = a * br + b * ar + c * dr - d * cr

    ct = a * cr - b * dr + c * ar + d * br

    dt = a * dr + b * cr - c * br + d * ar

    res = np.array([at, bt, ct, dt], dtype=quaternion1.dtype)

    norm = np.linalg.norm(res)
    res = res / norm
    return res


def axis_angle_to_quaternion(axis, angle):
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    s *= axis
    return np.concatenate([[c], s])


def r3_angle_to_quaternion(r3_angle):
    angle = np.linalg.norm(r3_angle)

    if angle > 1e-7:
        axis = (1 / angle) * r3_angle
        return axis_angle_to_quaternion(axis, angle)
    else:
        return np.array([1, 0, 0, 0], dtype=np.float32)


def quaternion_to_angle(quaternion):
    c = quaternion[0]
    if c > -1 and c < 1:
        angle = 2 * np.arccos(c)
        if angle > np.pi:
            angle -= 2 * np.pi
        s = np.sin(angle / 2)
        if s < 1e-7:
            return np.array([0, 0, 0], dtype=np.float32)
        axis = quaternion.copy()[1:]
        axis *= angle / s
        return axis
    else:
        return np.array([0, 0, 0], dtype=np.float32)


def quaternion_to_rotation_matrix(quaternion):
    a, b, c, d = quaternion
    aa, ab, ac, ad = a * quaternion
    bb, bc, bd = b * quaternion[1:]
    cc = c * c
    cd = c * d
    dd = d * d

    tmp = np.array([
        [aa + bb - cc - dd, 2 * (-ad + bc), 2 * (ac + bd)],
        [2 * (ad + bc), aa - bb + cc - dd, 2 * (-ab + cd)],
        [2 * (-ac + bd), 2 * (ab + cd), aa - bb - cc + dd]
    ], dtype=np.float32)
    return tmp


def random_unit_quaternion():
    q = np.random.normal(size=[4])
    norm = np.linalg.norm(q)
    if norm > 1e-7:
        q /= norm
        return q
    else:
        return random_unit_quaternion()


if __name__ == '__main__':
    r3_angle = np.array([0, 0, 0], dtype=np.float32)
    quaternion = r3_angle_to_quaternion(r3_angle)
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    tmp_coord = np.random.randn(10, 3)
    rotated_coord = (rotation_matrix @ tmp_coord.T).T
    assert np.allclose(tmp_coord, rotated_coord)
