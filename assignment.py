"""
Name : Prakhar Chauhan
Roll No: B23BB1032
Assignment 1


Pipeline:
    1.SIFT feature extraction
    2.Manual descriptor macth
    3.Lowe ratio test
    4.Hough transform
    5.Hough space cluster
    6.RANSAC affine transformation
    7.Least sq solution
    8.Geometric check
    9.Sequntial inlier suppression
    10.IOU
    11.final visualization"""

import math
import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np

TEMPLATE_PATH = "maggie.jpeg"
QUERY_PATH = "test.jpeg"

OUT_DIRS = "ans"

# SIFT para
SIFT_FEATURES = 2000

# Lowe ratio threshold
RATIO_THRESHOLD = 0.75
# Hough bin size
HOUGH_XY_BIN = 35.0
HOUGH_SCALE_BIN = 0.20
HOUGH_ANGLE_BIN = 20.0

# Minimum votes required for a Hough cluster
MIN_HOUGH_VOTES = 3

# RANSAC
RANSAC_ITERATIONS = 500
RANSAC_REPROJECTION_THRESHOLD = 8.0

# Minimum number of affine inliers
MIN_AFFINE_INLIERS = 3

# Final verification
MAX_MEAN_ERROR = 12.0
MAX_MEDIAN_ERROR = 12.0
MIN_INLIER_RATIO = 0.35
MIN_AFFINE_DETERMINANT = 0.01
MAX_AFFINE_DETERMINANT = 25.0
MIN_AFFINE_SCALE = 0.1
MAX_AFFINE_SCALE = 5.0
MAX_AFFINE_ASPECT_RATIO = 3.0

# NMS threshold
NMS_IOU_THRESHOLD = 0.30


def ensure_():
    os.makedirs(OUT_DIRS, exist_ok=True)


def save_fig(name):
    path = os.path.join(OUT_DIRS, name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved", path)


# Load imgs


def load_img():
    template = cv2.imread(TEMPLATE_PATH)
    query = cv2.imread(QUERY_PATH)

    if template is None:
        raise FileNotFoundError(f"Could not load template image: {TEMPLATE_PATH}")
    if query is None:
        raise FileNotFoundError(f"Could not load query image: {QUERY_PATH}")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

    return template, template_gray, query, query_gray


def extract_sift(img_gray):
    # 128 dimensional descriptors
    #  return keypoint , descriptos
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)

    keypts, descp = sift.detectAndCompute(img_gray, None)

    return keypts, descp


# Visualize SIFT keypoint


def visualize_sift(template, query, kp_template, kp_query):
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

    query_rgb = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(template_rgb)

    pts = np.array([kp.pt for kp in kp_template])

    if len(pts) > 0:
        plt.scatter(pts[:, 0], pts[:, 1], s=8)
    plt.title(f"Template SIFT Keypts ({len(kp_template)})")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(query_rgb)
    pts = np.array([kp.pt for kp in kp_query])
    if len(pts) > 0:
        plt.scatter(pts[:, 0], pts[:, 1], s=4)
    plt.title(f"Query SIFT keypoint({len(kp_query)})")
    plt.axis("off")
    plt.tight_layout()
    save_fig("01_sift_keypoints.png")

    plt.close()


def euclidean_distance(desc1, desc2):
    """
    d(x,y) = sqrt(sum(x_1-y_1)^2)
    """
    diff = desc1.astype(np.float32) - desc2.astype(np.float32)

    return np.sqrt(np.sum(diff * diff))


def manual_ratio_match(kp_template, desc_template, kp_query, desc_query):
    """
    Match query feature against template feature
    uing lowe ratio:
    d1 / d2 <threshold
    where d1 = nearest template descriptor distance
    d2 = second nearest distance
    """

    matches = []
    if desc_template is None or desc_query is None:
        return matches

    for query_idx in range(len(desc_query)):
        query_desc = desc_query[query_idx]
        distances = []

        for template_idx in range(len(desc_template)):
            distance = euclidean_distance(query_desc, desc_template[template_idx])
            distances.append((distance, template_idx))

        if len(distances) < 2:
            continue

        distances.sort(key=lambda x: x[0])
        d1, template_idx = distances[0]
        d2, _ = distances[1]

        # Lowes ratio test:
        if d2 == 0:
            continue
        ratio = d1 / d2

        if ratio < RATIO_THRESHOLD:
            matches.append(
                {
                    "query_idx": query_idx,
                    "template_idx": template_idx,
                    "distance": d1,
                    "ratio": ratio,
                }
            )
    return matches


def visualize_matches(template, query, kp_temp, kp_query, matches, filename, title):

    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

    query_rgb = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
    h1, w1 = template_rgb.shape[:2]
    h2, w2 = query_rgb.shape[:2]

    canvas_height = max(h1, h2)
    canvas_width = w1 + w2

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    canvas[:h1, :w1] = template_rgb
    canvas[:h2, w1 : w1 + w2] = query_rgb

    plt.figure(figsize=(18, 9))

    plt.imshow(canvas)

    display_matches = matches[:150]

    for mat in display_matches:
        t_idx = mat["template_idx"]
        q_idx = mat["query_idx"]

        x1, y1 = kp_temp[t_idx].pt
        x2, y2 = kp_query[q_idx].pt

        x2 += w1

        plt.plot([x1, x2], [y1, y2], linewidth=0.5)
        plt.scatter([x1, x2], [y1, y2], s=5)

    plt.axvline(x=w1, linewidth=2)
    plt.title(title + f"Showing {len(display_matches)} matches")
    plt.axis("off")
    plt.tight_layout()
    save_fig(filename)
    plt.close()


def get_match_points(matches, kp_template, kp_query):
    """
    Convert match dictionaries into point arrays.
    """

    template_points = []
    query_points = []

    for match in matches:
        t_idx = match["template_idx"]
        q_idx = match["query_idx"]

        template_points.append(kp_template[t_idx].pt)

        query_points.append(kp_query[q_idx].pt)

    return (
        np.asarray(template_points, dtype=np.float64),
        np.asarray(query_points, dtype=np.float64),
    )


def normalize_angle(angle):
    """
    Convert angle to [0, 360).
    """

    angle = angle % 360.0

    if angle < 0:
        angle += 360.0

    return angle


def hough_bin_key(center_x, center_y, scale, angle):
    """
    Convert predicted object parameters into a 4-D discrete bin.

    Dimensions:

        X
        Y
        Scale
        Orientation
    """

    x_bin = int(math.floor(center_x / HOUGH_XY_BIN))

    y_bin = int(math.floor(center_y / HOUGH_XY_BIN))

    # Log-scale is more stable for scale changes.
    scale_bin = int(math.floor(math.log(max(scale, 1e-6)) / HOUGH_SCALE_BIN))

    angle = normalize_angle(angle)

    angle_bin = int(math.floor(angle / HOUGH_ANGLE_BIN))

    return (x_bin, y_bin, scale_bin, angle_bin)


def rotate_vector(vector, angle_degrees):
    """
    Rotate 2-D vector by angle.
    """

    theta = np.deg2rad(angle_degrees)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return R @ vector


def generalized_hough_voting(matches, kp_template, kp_query, template_shape):
    template_height, template_width = template_shape[:2]

    template_center = np.array([template_width / 2.0, template_height / 2.0])

    votes = defaultdict(list)

    vote_records = []

    for match in matches:
        t_idx = match["template_idx"]
        q_idx = match["query_idx"]

        t_kp = kp_template[t_idx]
        q_kp = kp_query[q_idx]

        # -----------------------------------------------------
        # Scale ratio
        # -----------------------------------------------------

        if t_kp.size <= 0:
            continue

        scale = q_kp.size / t_kp.size

        if scale <= 0:
            continue

        # -----------------------------------------------------
        # Orientation difference
        # -----------------------------------------------------

        angle_difference = normalize_angle(q_kp.angle - t_kp.angle)

        # -----------------------------------------------------
        # Relative feature position in template
        # -----------------------------------------------------

        template_feature = np.array(t_kp.pt)

        relative_vector = template_feature - template_center

        # -----------------------------------------------------
        # Rotate + scale relative vector
        # -----------------------------------------------------

        transformed_vector = scale * rotate_vector(relative_vector, angle_difference)

        # -----------------------------------------------------
        # Predicted object center
        # -----------------------------------------------------

        query_feature = np.array(q_kp.pt)

        predicted_center = query_feature - transformed_vector

        predicted_x = predicted_center[0]
        predicted_y = predicted_center[1]

        # -----------------------------------------------------
        # Create 4-D Hough bin
        # -----------------------------------------------------

        key = hough_bin_key(predicted_x, predicted_y, scale, angle_difference)

        votes[key].append(match)

        vote_records.append(
            {
                "x": predicted_x,
                "y": predicted_y,
                "scale": scale,
                "angle": angle_difference,
                "match": match,
                "bin": key,
            }
        )

    return votes, vote_records


def visualize_hough_votes(query, vote_records, clusters):
    """
    Visualize predicted object centers.

    Dense regions indicate likely object instances.
    """

    query_rgb = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))

    plt.imshow(query_rgb)

    if len(vote_records) > 0:
        xs = [record["x"] for record in vote_records]

        ys = [record["y"] for record in vote_records]

        plt.scatter(xs, ys, s=12, alpha=0.45, label="Individual votes")

    for index, cluster in enumerate(clusters):
        plt.scatter(*cluster["center"], s=120, marker="x", linewidths=3, label=f"Cluster {index + 1}")

    plt.title("4-D Generalized Hough Transform: Predicted Object Centers")

    plt.axis("off")

    plt.tight_layout()

    save_fig("03_hough_votes.png")

    plt.close()


def find_hough_clusters(votes, vote_records):
    """
    Find bins having enough votes.

    Each strong bin is considered a candidate object instance.
    """

    clusters = {}
    records_by_match = {id(record["match"]): record for record in vote_records}
    angle_bins = int(360 / HOUGH_ANGLE_BIN)

    for x_bin, y_bin, scale_bin, angle_bin in votes:
        matches = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for ds in (-1, 0, 1):
                    for da in (-1, 0, 1):
                        matches.extend(votes.get((x_bin + dx, y_bin + dy, scale_bin + ds, (angle_bin + da) % angle_bins), []))
        if len(matches) < MIN_HOUGH_VOTES:
            continue
        records = [records_by_match[id(match)] for match in matches]
        angles = np.deg2rad([record["angle"] for record in records])
        clusters[tuple(sorted((m["query_idx"], m["template_idx"]) for m in matches))] = {
            "bin": (x_bin, y_bin, scale_bin, angle_bin), "matches": matches, "votes": len(matches),
            "center": (float(np.mean([record["x"] for record in records])), float(np.mean([record["y"] for record in records]))),
            "scale": float(np.median([record["scale"] for record in records])),
            "angle": normalize_angle(np.rad2deg(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))),
        }

    # Strongest clusters first
    clusters = sorted(clusters.values(), key=lambda x: x["votes"], reverse=True)

    return clusters


def are_collinear(points, epsilon=1e-6):

    if len(points) < 3:
        return True

    p1 = points[0]
    p2 = points[1]
    p3 = points[2]

    v1 = p2 - p1
    v2 = p3 - p1

    cross_product = v1[0] * v2[1] - v1[1] * v2[0]

    return abs(cross_product) < epsilon


def solve_affine_from_three(src_points, dst_points):
    """
    Calculate affine transformation from 3 point pairs.

    Affine transformation:

        x' = a*x + b*y + c
        y' = d*x + e*y + f

    Matrix:

        [x']   [a b c] [x]
        [y'] = [d e f] [y]
                       [1]

    We solve the 6 unknown coefficients.

    This function is used inside RANSAC.
    """

    if len(src_points) != 3:
        return None

    if len(dst_points) != 3:
        return None

    # Reject collinear source points
    if are_collinear(src_points):
        return None

    # Reject collinear destination points
    if are_collinear(dst_points):
        return None

    A = []
    b = []

    for i in range(3):
        x, y = src_points[i]
        X, Y = dst_points[i]

        A.append([x, y, 1, 0, 0, 0])

        b.append(X)

        A.append([0, 0, 0, x, y, 1])

        b.append(Y)

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    try:
        # Solve Ax = b
        coefficients = np.linalg.solve(A, b)

    except np.linalg.LinAlgError:
        return None

    affine = np.array(
        [
            [coefficients[0], coefficients[1], coefficients[2]],
            [coefficients[3], coefficients[4], coefficients[5]],
        ]
    )

    return affine


def affine_least_squares(src_points, dst_points):
    """
    Estimate affine transformation using ALL available
    inliers.

    We solve:

        A x = b

    using the least-squares solution:

        x = (A^T A)^(-1) A^T b

    Instead of explicitly computing the inverse, NumPy's
    lstsq() is used for numerical stability.

    This is still our mathematical least-squares solution,
    not a high-level geometric API.
    """

    if len(src_points) < 3:
        return None

    A = []
    b = []

    for i in range(len(src_points)):
        x, y = src_points[i]
        X, Y = dst_points[i]

        A.append([x, y, 1, 0, 0, 0])

        b.append(X)

        A.append([0, 0, 0, x, y, 1])

        b.append(Y)

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    try:
        coefficients, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)

    except np.linalg.LinAlgError:
        return None

    if rank < 6 and len(src_points) >= 3:
        # For exactly 3 non-collinear points rank should be 6.
        # With more points, rank should also normally be 6.
        return None

    affine = np.array(
        [
            [coefficients[0], coefficients[1], coefficients[2]],
            [coefficients[3], coefficients[4], coefficients[5]],
        ]
    )

    return affine


# 13. APPLY AFFINE TRANSFORMATION


def transform_points(points, affine):
    """
    Apply affine transformation manually.

        [X]   [a b c] [x]
        [Y] = [d e f] [y]
                         [1]
    """

    if affine is None:
        return None

    points = np.asarray(points)

    ones = np.ones((len(points), 1))

    homogeneous = np.hstack([points, ones])

    transformed = homogeneous @ affine.T

    return transformed


# 14. RANSAC


def ransac_affine(
    src_points,
    dst_points,
    iterations=RANSAC_ITERATIONS,
    threshold=RANSAC_REPROJECTION_THRESHOLD,
):
    """
    Robustly estimate affine transformation.

    RANSAC procedure:

        1. Randomly select 3 correspondences.
        2. Estimate affine transformation.
        3. Transform all source points.
        4. Calculate reprojection error.
        5. Count inliers.
        6. Keep model with maximum inliers.
        7. Re-estimate using least squares on inliers.
    """

    n = len(src_points)

    if n < 3:
        return None, []

    best_affine = None
    best_inliers = []

    rng = np.random.default_rng(42)

    for iteration in range(iterations):
        # Randomly select 3 correspondences
        sample_indices = rng.choice(n, size=3, replace=False)

        sample_src = src_points[sample_indices]

        sample_dst = dst_points[sample_indices]

        # Estimate affine model
        affine = solve_affine_from_three(sample_src, sample_dst)

        if affine is None:
            continue

        # Transform every source point
        predicted = transform_points(src_points, affine)

        # Reprojection error
        errors = np.sqrt(np.sum((predicted - dst_points) ** 2, axis=1))

        inlier_indices = np.where(errors < threshold)[0]

        if len(inlier_indices) > len(best_inliers):
            best_inliers = inlier_indices.tolist()
            best_affine = affine

    # Need at least 3 inliers
    if best_affine is None:
        return None, []

    if len(best_inliers) < 3:
        return None, []

    # ---------------------------------------------------------
    # Least-Squares refinement
    # ---------------------------------------------------------

    refined_affine = affine_least_squares(
        src_points[best_inliers], dst_points[best_inliers]
    )

    if refined_affine is None:
        return None, []

    # Recalculate inliers after refinement
    predicted = transform_points(src_points, refined_affine)

    errors = np.sqrt(np.sum((predicted - dst_points) ** 2, axis=1))

    final_inliers = np.where(errors < threshold)[0].tolist()

    return refined_affine, final_inliers


# 15. GEOMETRIC SANITY CHECKS


def affine_sanity_check(affine, template_shape, scene_shape):
    """
    Reject geometrically impossible affine transformations.

    Checks:

        1. determinant
        2. scale
        3. bounding-box size
        4. finite values
    """

    if affine is None:
        return False

    if not np.all(np.isfinite(affine)):
        return False

    linear = affine[:, :2]

    determinant = np.linalg.det(linear)

    # determinant approximately represents area scaling
    if abs(determinant) < MIN_AFFINE_DETERMINANT:
        return False

    if abs(determinant) > MAX_AFFINE_DETERMINANT:
        return False

    # Estimate scale from column lengths
    scale_x = np.linalg.norm(linear[:, 0])

    scale_y = np.linalg.norm(linear[:, 1])

    if scale_x < MIN_AFFINE_SCALE or scale_y < MIN_AFFINE_SCALE:
        return False

    if scale_x > MAX_AFFINE_SCALE or scale_y > MAX_AFFINE_SCALE:
        return False

    if max(scale_x, scale_y) / min(scale_x, scale_y) > MAX_AFFINE_ASPECT_RATIO:
        return False

    return True


# 16. PROJECT TEMPLATE CORNERS


def project_template_corners(affine, template_shape):
    """
    Project four template corners into the query image.
    """

    h, w = template_shape[:2]

    corners = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64
    )

    projected = transform_points(corners, affine)

    return projected


# 17. CHECK POLYGON VALIDITY


def polygon_area(points):
    """
    Shoelace formula.
    """

    area = 0.0

    for i in range(len(points)):
        x1, y1 = points[i]

        x2, y2 = points[(i + 1) % len(points)]

        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def polygon_is_reasonable(corners, scene_shape):
    """
    Check projected template polygon.

    Reject:

        - tiny polygons
        - enormous polygons
        - NaN/Inf
    """

    if corners is None:
        return False

    if not np.all(np.isfinite(corners)):
        return False

    area = polygon_area(corners)

    scene_h, scene_w = scene_shape[:2]

    scene_area = scene_h * scene_w

    # Avoid extremely tiny detections
    if area < 0.005 * scene_area:
        return False

    # Avoid absurdly large detections
    if area > 1.5 * scene_area:
        return False

    return True


# 18. CALCULATE REPROJECTION ERROR


def reprojection_statistics(src_points, dst_points, affine):
    """
    Calculate errors between transformed template points
    and observed scene points.
    """

    predicted = transform_points(src_points, affine)

    errors = np.sqrt(np.sum((predicted - dst_points) ** 2, axis=1))

    if len(errors) == 0:
        return {"mean": float("inf"), "median": float("inf"), "max": float("inf")}

    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
    }


# 19. BOUNDING BOX FROM POLYGON


def polygon_to_bbox(polygon, image_shape):
    """
    Convert projected quadrilateral into an axis-aligned
    bounding box.

    Format:

        [x1, y1, x2, y2]
    """

    h, w = image_shape[:2]

    xs = polygon[:, 0]
    ys = polygon[:, 1]

    x1 = max(0, int(np.floor(np.min(xs))))

    y1 = max(0, int(np.floor(np.min(ys))))

    x2 = min(w - 1, int(np.ceil(np.max(xs))))

    y2 = min(h - 1, int(np.ceil(np.max(ys))))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


# 20. IOU FROM SCRATCH


def calculate_iou(box_a, box_b):
    """
    Intersection over Union.

        IoU = Area(Intersection)
              ------------------
              Area(Union)
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection coordinates
    ix1 = max(ax1, bx1)

    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)

    iy2 = min(ay2, by2)

    # Intersection dimensions
    iw = max(0, ix2 - ix1)

    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    # Areas
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)

    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# 21. NMS FROM SCRATCH


def non_maximum_suppression(detections, iou_threshold=NMS_IOU_THRESHOLD):
    """
    Standard greedy NMS.

    Steps:

        1. Sort detections by score.
        2. Keep highest scoring box.
        3. Remove boxes having high IoU with it.
        4. Repeat.

    No OpenCV NMS function is used.
    """

    if len(detections) == 0:
        return []

    remaining = sorted(detections, key=lambda d: d["score"], reverse=True)

    selected = []

    while len(remaining) > 0:
        best = remaining.pop(0)

        selected.append(best)

        survivors = []

        for detection in remaining:
            iou = calculate_iou(best["bbox"], detection["bbox"])

            if iou < iou_threshold:
                survivors.append(detection)

        remaining = survivors

    return selected


# 22. VISUALIZE IoU MATRIX


def visualize_iou_matrix(detections):
    """
    Display IoU matrix before NMS.

    This is useful for the report to show how duplicate
    detections are suppressed.
    """

    n = len(detections)

    if n == 0:
        return

    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i, j] = calculate_iou(detections[i]["bbox"], detections[j]["bbox"])

    plt.figure(figsize=(7, 6))

    plt.imshow(matrix, interpolation="nearest")

    plt.colorbar(label="IoU")

    plt.xlabel("Detection")

    plt.ylabel("Detection")

    plt.title("IoU Matrix Before Non-Maximum Suppression")

    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()

    save_fig("06_iou_matrix.png")

    plt.close()


# 23. DRAW DETECTIONS


def draw_detections(image, detections, title, filename, show_polygon=True):
    """
    Draw final detections.

    Green/normal bounding polygons are produced using the
    calculated affine transformation.
    """

    output = image.copy()

    for idx, detection in enumerate(detections):
        polygon = detection["polygon"]
        bbox = detection["bbox"]

        if show_polygon:
            pts = np.round(polygon).astype(np.int32)

            cv2.polylines(output, [pts], True, (0, 255, 0), 3)


        x1, y1, x2, y2 = bbox

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)

        label = (
            f"Maggi {idx + 1} "
            f"| inliers={detection['inliers']} "
            f"| err={detection['mean_error']:.1f}"
        )

        cv2.putText(
            output,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))

    plt.imshow(output_rgb)

    plt.title(title)

    plt.axis("off")

    plt.tight_layout()

    save_fig(filename)

    plt.close()

    return output


# 24. VISUALIZE Hough CLUSTERS


def visualize_hough_clusters(query, clusters):
    """
    Show where strong Hough clusters occur.
    """

    query_rgb = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 9))

    plt.imshow(query_rgb)

    colors = ["red", "blue", "lime", "orange", "magenta", "cyan", "yellow"]

    for cluster_id, cluster in enumerate(clusters):
        color = colors[cluster_id % len(colors)]

        x, y = cluster["center"]
        plt.scatter(x, y, color=color, s=140, marker="x")
        plt.text(x, y, f"#{cluster_id + 1}: {cluster['votes']} votes\ns={cluster['scale']:.2f}, a={cluster['angle']:.0f}°", color=color, fontsize=9)

    plt.title("Strong Hough Clusters")

    plt.axis("off")

    plt.tight_layout()

    save_fig("04_hough_clusters.png")

    plt.close()


# 25. VERIFY A Hough CLUSTER


def verify_hough_cluster(cluster, kp_template, kp_query, template_shape, query_shape):
    """
    Perform complete geometric verification on one Hough cluster.

    Steps:

        Hough cluster
             ↓
        point correspondences
             ↓
        RANSAC affine
             ↓
        least-squares refinement
             ↓
        geometric checks
             ↓
        reprojection error
             ↓
        projected template corners
             ↓
        bounding box
    """

    matches = cluster["matches"]

    if len(matches) < MIN_AFFINE_INLIERS:
        return None

    template_points, query_points = get_match_points(matches, kp_template, kp_query)

    # ---------------------------------------------------------
    # RANSAC
    # ---------------------------------------------------------

    affine, inlier_indices = ransac_affine(template_points, query_points)

    if affine is None:
        return None

    if len(inlier_indices) < MIN_AFFINE_INLIERS:
        return None

    # ---------------------------------------------------------
    # Geometric sanity check
    # ---------------------------------------------------------

    if not affine_sanity_check(affine, template_shape, query_shape):
        return None

    # ---------------------------------------------------------
    # Reprojection error
    # ---------------------------------------------------------

    stats = reprojection_statistics(
        template_points[inlier_indices], query_points[inlier_indices], affine
    )

    if stats["mean"] > MAX_MEAN_ERROR:
        return None

    if stats["median"] > MAX_MEDIAN_ERROR:
        return None


    inlier_ratio = len(inlier_indices) / len(matches)

    if inlier_ratio < MIN_INLIER_RATIO:
        return None


    polygon = project_template_corners(affine, template_shape)

    if not polygon_is_reasonable(polygon, query_shape):
        return None

    bbox = polygon_to_bbox(polygon, query_shape)

    if bbox is None:
        return None

    score = len(inlier_indices) * inlier_ratio / (1.0 + stats["mean"])

    return {
        "affine": affine,
        "polygon": polygon,
        "bbox": bbox,
        "inliers": len(inlier_indices),
        "inlier_ratio": inlier_ratio,
        "mean_error": stats["mean"],
        "median_error": stats["median"],
        "scale": cluster["scale"],
        "angle": cluster["angle"],
        "determinant": float(np.linalg.det(affine[:, :2])),
        "score": score,
        "matches": matches,
        "inlier_indices": inlier_indices,
    }


# 26. SEQUENTIAL INLIER SUBTRACTION


def greedy_detection(matches, kp_template, kp_query, template_shape, query_shape):
    """
    Greedy multi-instance detection.

    Main idea:

        1. Build Hough space.
        2. Find strongest cluster.
        3. Geometrically verify it.
        4. Remove its inlier matches.
        5. Re-run Hough voting.
        6. Find next object.
        7. Repeat.

    This is the sequential inlier subtraction requested
    in the assignment.
    """

    remaining_matches = list(matches)

    detections = []

    max_objects = 10

    for iteration in range(max_objects):
        if len(remaining_matches) < 3:
            break

        print(f"\n========== Greedy Iteration {iteration + 1} ==========")

        # -----------------------------------------------------
        # Hough voting on remaining matches
        # -----------------------------------------------------

        votes, vote_records = generalized_hough_voting(
            remaining_matches, kp_template, kp_query, template_shape
        )

        clusters = find_hough_clusters(votes, vote_records)

        print("Remaining matches:", len(remaining_matches))

        print("Strong Hough clusters:", len(clusters))

        if len(clusters) == 0:
            break

        found_detection = False

        # -----------------------------------------------------
        # Try strongest clusters first
        # -----------------------------------------------------

        for cluster in clusters:
            detection = verify_hough_cluster(
                cluster, kp_template, kp_query, template_shape, query_shape
            )

            if detection is None:
                continue

            print("Valid detection found:")

            print("  Hough votes:", cluster["votes"])

            print("  Affine inliers:", detection["inliers"])

            print("  Inlier ratio:", round(detection["inlier_ratio"], 3))

            print("  Mean reprojection error:", round(detection["mean_error"], 3))
            print("  Scale:", round(detection["scale"], 3))
            print("  Orientation:", round(detection["angle"], 1))

            # -------------------------------------------------
            # Add detection
            # -------------------------------------------------

            detections.append(detection)

            # -------------------------------------------------
            # Remove matched inlier correspondences
            # -------------------------------------------------

            inlier_match_indices = set(
                cluster["matches"][i]["query_idx"] for i in detection["inlier_indices"]
            )

            new_remaining = []

            for match in remaining_matches:
                if match["query_idx"] not in inlier_match_indices:
                    new_remaining.append(match)

            remaining_matches = new_remaining

            found_detection = True

            break

        # -----------------------------------------------------
        # No valid cluster found
        # -----------------------------------------------------

        if not found_detection:
            print("No geometrically valid cluster remains.")

            break

    return detections


# 27. DRAW MATCHES USED BY ONE DETECTION


def visualize_detection_matches(query, kp_query, detection, detection_id):
    """
    Show only the query-side inlier points belonging to
    a detected object.
    """

    image = query.copy()

    for match_index in detection["inlier_indices"]:
        match = detection["matches"][match_index]

        q_idx = match["query_idx"]

        x, y = kp_query[q_idx].pt

        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))

    plt.imshow(image_rgb)

    plt.title(f"Verified SIFT Inliers - Detection {detection_id}")

    plt.axis("off")

    plt.tight_layout()

    save_fig(f"05_verified_inliers_{detection_id}.png")

    plt.close()


def print_detections(detections, title):
    """
    Print numerical information for the report.
    """

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    print("Number of detections:", len(detections))

    for i, detection in enumerate(detections):
        print(f"\nDetection {i + 1}")

        print("Bounding box:", detection["bbox"])

        print("Inliers:", detection["inliers"])

        print("Inlier ratio:", round(detection["inlier_ratio"], 3))

        print("Mean error:", round(detection["mean_error"], 3))

        print("Median error:", round(detection["median_error"], 3))

        print("Scale:", round(detection["scale"], 3))

        print("Orientation:", round(detection["angle"], 1))

        print("Determinant:", round(detection["determinant"], 4))

        print("Score:", round(detection["score"], 3))

        print("Affine matrix:")

        print(detection["affine"])


def main():

    ensure_()

    print("=" * 70)
    print("MULTI-INSTANCE OBJECT RECOGNITION")
    print("MAGGI NOODLES PACKET")
    print("=" * 70)

    template, template_gray, query, query_gray = load_img()

    print("\nDATASET")
    print("Template image:", TEMPLATE_PATH)
    print("Query image:", QUERY_PATH)

    print("\nTemplate size:", template.shape)

    print("Query size:", query.shape)

    print("\n[1] Extracting SIFT features...")

    kp_template, des_template = extract_sift(template_gray)

    kp_query, des_query = extract_sift(query_gray)

    print("Template keypoints:", len(kp_template))

    print("Query keypoints:", len(kp_query))

    if des_template is not None:
        print("Template descriptor shape:", des_template.shape)

    if des_query is not None:
        print("Query descriptor shape:", des_query.shape)

    visualize_sift(template, query, kp_template, kp_query)

    print("\n[2] Performing manual descriptor matching...")

    matches = manual_ratio_match(kp_template, des_template, kp_query, des_query)

    distances = [match["distance"] for match in matches]
    ratios = [match["ratio"] for match in matches]
    print("Total descriptor comparisons:", len(des_query) * len(des_template))
    print("Ratio-test matches:", len(matches))
    if matches:
        print("Distance min/median/max:", *(round(value, 2) for value in (min(distances), np.median(distances), max(distances))))
        print("Median Lowe ratio:", round(float(np.median(ratios)), 3))

    visualize_matches(
        template,
        query,
        kp_template,
        kp_query,
        matches,
        "02_ratio_matches.png",
        "Manual SIFT Matching + Lowe Ratio Test",
    )

    print("\n[3] Performing 4-D Generalized Hough Transform...")

    votes, vote_records = generalized_hough_voting(
        matches, kp_template, kp_query, template.shape
    )

    clusters = find_hough_clusters(votes, vote_records)

    print("Hough occupied bins:", len(votes))
    print("Strong candidate clusters:", len(clusters))
    for index, record in enumerate(vote_records[:5], 1):
        print(f"Match {index}: template={kp_template[record['match']['template_idx']].pt}, query={kp_query[record['match']['query_idx']].pt}, scale={record['scale']:.3f}, angle={record['angle']:.1f}, center=({record['x']:.1f}, {record['y']:.1f}), bin={record['bin']}")

    visualize_hough_votes(query, vote_records, clusters)
    visualize_hough_clusters(query, clusters)

    print("\n[4] Running RANSAC + Least Squares + Sequential Inlier Subtraction...")

    detections = greedy_detection(
        matches, kp_template, kp_query, template.shape, query.shape
    )

    print_detections(detections, "DETECTIONS BEFORE NMS")

    draw_detections(query, detections, "MAGGI DETECTIONS BEFORE NMS", "06_before_nms.png")

    for i, detection in enumerate(detections):
        visualize_detection_matches(query, kp_query, detection, i + 1)

    visualize_iou_matrix(detections)

    print("\n[5] Performing manual IoU Non-Maximum Suppression...")

    final_detections = non_maximum_suppression(detections, NMS_IOU_THRESHOLD)

    print_detections(final_detections, "FINAL DETECTIONS AFTER NMS")

    final_image = draw_detections(
        query,
        final_detections,
        "FINAL MULTI-INSTANCE MAGGI DETECTION",
        "07_final_detection.png",
        show_polygon=True,
    )

    cv2.imwrite(os.path.join(OUT_DIRS, "final_detection.jpg"), final_image)

    print("Pipeline complete.")

    print("Results saved in:", OUT_DIRS)


if __name__ == "__main__":
    main()
