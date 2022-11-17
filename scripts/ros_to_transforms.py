import numpy as np
import cv2
import json


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))



def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_sharpness(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = variance_of_laplacian(gray)


    return sharpness


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2
        ]
    ])


if __name__ == "__main__":
    position = [-0.0170282, 0.001679, 0.032355]
    orientation = [0.0157856, 0.0121887, 0.00470192, 0.99979]

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])

    sharpness = get_sharpness('0001.jpeg')
    
    # camera intrinsics can be hard-coded later
    out = {
        "camera_angle_x": 1,
        "camera_angle_y": 1,
        "fl_x": 1,
        "fl_y": 1,
        "k1": 1,
        "k2": 1,
        "p1": 1,
        "p2": 1,
        "cx": 1,
        "cy": 1,
        "w": 1,
        "h": 1,
        "aabb_scale": 1,
        "frames": [],
    }

    q_vec = np.array(orientation)
    t_vec = np.array(position)
    R = qvec2rotmat(-q_vec)  # Get rotation
    t = t_vec.reshape([3, 1])

    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)

    print(m)

    c2w = np.linalg.inv(m)

    print(c2w)

    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    transform_matrix = np.matmul(c2w, flip_mat)

    print(transform_matrix)
    frame = {"file_path": '0001.jpeg', "sharpness": sharpness, "transform_matrix": transform_matrix.tolist()}
    out["frames"].append(frame)
    OUT_PATH = 'transforms.json'

    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)







