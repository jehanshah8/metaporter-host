import numpy as np
import cv2
import json

keep_colmap_coords = False


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


def closest_point_2_lines(oa, da, ob,
                          db):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


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


def create_c2w_matrix(position, orientation, img_path, c2w_out, up_file):
    with open(up_file) as upfile:
        up_dict = json.load(upfile)

    up = up_dict["up"]

    with open(c2w_out) as c2w_file:
        out = json.load(c2w_file)

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    sharpness = get_sharpness(img_path)

    q_vec = np.array(orientation)
    t_vec = np.array(position)
    R = qvec2rotmat(-q_vec)  # Get rotation
    t = t_vec.reshape([3, 1])

    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)



    c2w = np.linalg.inv(m)



    if not keep_colmap_coords:
        c2w[0:3, 2] *= -1  # flip the y and z axis
        c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
        c2w[2, :] *= -1  # flip whole world upside down

        up += c2w[0:3, 1]

    print(c2w)
    frame = {"file_path": '0001.jpeg', "sharpness": sharpness, "transform_matrix": c2w.tolist()}
    out["frames"].append(frame)

    up_dict = {"up": up.tolist()}

    with open(up_file, "w") as upfile:
        json.dump(up_dict, upfile, indent=2)

    with open(c2w_out, "w") as c2wfile:
        json.dump(out, c2wfile, indent=2)

    return up


def write_to_transforms(up,
                        OUT_PATH):  # This Function preforms additional Transformations according to NVIDIAS colmap2Nerf

    nframes = len(out["frames"])
    if keep_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat)  # flip cameras (it just works)
    else:
        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis

        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3, :]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3, :]
                p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])

                if w > 0.00001:
                    totp += p * w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp)  # the cameras are looking at totp
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)


if __name__ == "__main__":
    out = {
        "camera_angle_x": 1,
        "camera_angle_y": 1,
        "fl_x": 1,
        "fl_y": 1,
        "k1": 1,
        "k2": 70,
        "p1": 1,
        "p2": 1,
        "cx": 1,
        "cy": 1,
        "w": 1,
        "h": 1,
        "aabb_scale": 1,
        "frames": [],
    }

    up = np.zeros(3)
    c2w_out = 'c2w.json'

    up_file = 'up.json'

    with open(c2w_out, "w") as c2wfile:
        json.dump(out, c2wfile, indent=2)

    up_dict = {"up": up.tolist()}

    with open(up_file, "w") as upfile:
        json.dump(up_dict, upfile, indent=2)



    position = [-0.0170282, 0.001679, 0.032355]
    orientation = [0.0157856, 0.0121887, 0.00470192, 0.99979]

    create_c2w_matrix(position, orientation, '0001.jpeg', c2w_out, up_file)
    create_c2w_matrix(position, orientation, '0001.jpeg', c2w_out, up_file)

    position = [-0.0170282, 0.101679, 0.032355]
    orientation = [0.0197856, 0.0121887, 0.01470192, 0.77979]

    create_c2w_matrix(position, orientation, '0001.jpeg', c2w_out, up_file)

    # write_to_transforms(up, OUT_PATH='transforms.json')

    # transform_matrix = np.matmul(c2w, flip_mat)

    # OUT_PATH = 'transforms.json'

    # with open(OUT_PATH, "w") as outfile:
    # json.dump(out, outfile, indent=2)
