"""Small dependency-free geometry helpers."""

import math


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle):
    if not math.isfinite(angle):
        raise ValueError("nonfinite angle")
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def speed_2d(vx, vy):
    return math.hypot(vx, vy)


def project_polyline(points, x, y):
    """Nearest bounded segment projection; no extrapolation past map coverage."""
    best, along = None, 0.0
    for index in range(len(points) - 1):
        ax, ay = points[index][:2]
        bx, by = points[index + 1][:2]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        raw_ratio = ((x - ax) * dx + (y - ay) * dy) / (length * length)
        ratio = clamp(raw_ratio, 0.0, 1.0)
        px, py = ax + ratio * dx, ay + ratio * dy
        distance = math.hypot(x - px, y - py)
        if best is None or distance < best["distance"]:
            best = {"s": along + ratio * length, "distance": distance,
                    "index": index, "ratio": ratio, "raw_ratio": raw_ratio,
                    "point": (px, py), "heading": math.atan2(dy, dx)}
        along += length
    return best


def world_to_ego(ego_x, ego_y, heading, target_x, target_y):
    dx = target_x - ego_x
    dy = target_y - ego_y
    c = math.cos(heading)
    s = math.sin(heading)
    longitudinal = c * dx + s * dy
    lateral = -s * dx + c * dy
    return longitudinal, lateral


def calculate_ttc(longitudinal_distance, ego_speed, target_vx, target_vy, heading):
    target_forward_speed = math.cos(heading) * target_vx + math.sin(heading) * target_vy
    closing_speed = ego_speed - target_forward_speed
    if longitudinal_distance <= 0.0 or closing_speed <= 0.05:
        return -1.0, closing_speed
    return longitudinal_distance / closing_speed, closing_speed


def nearest_path_error(points, x, y, heading):
    """Return signed lateral error and heading error for a polyline."""
    if not points or len(points) < 2:
        return 0.0, 0.0
    best = None
    for index in range(len(points) - 1):
        x1, y1 = points[index][0], points[index][1]
        x2, y2 = points[index + 1][0], points[index + 1][1]
        vx, vy = x2 - x1, y2 - y1
        length2 = vx * vx + vy * vy
        if length2 < 1e-8:
            continue
        ratio = clamp(((x - x1) * vx + (y - y1) * vy) / length2, 0.0, 1.0)
        px, py = x1 + ratio * vx, y1 + ratio * vy
        dx, dy = x - px, y - py
        distance2 = dx * dx + dy * dy
        if best is None or distance2 < best[0]:
            cross = vx * (y - py) - vy * (x - px)
            signed = math.sqrt(distance2) if cross >= 0.0 else -math.sqrt(distance2)
            path_heading = math.atan2(vy, vx)
            best = (distance2, signed, normalize_angle(path_heading - heading))
    if best is None:
        return 0.0, 0.0
    return best[1], best[2]
