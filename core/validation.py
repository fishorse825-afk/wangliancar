"""Validate contracts at each member boundary before anything can be sent."""

import math
import time

from core.interfaces import DecisionMode, DecisionTarget, Trajectory, ControlOut


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def current(value, source=None):
    if (type(value.valid) is not bool or not value.valid
            or type(value.frame_id) is not int or value.frame_id < 0
            or type(value.timestamp) is not int or value.timestamp < 0
            or not number(value.valid_until) or time.monotonic() >= value.valid_until):
        return False
    return source is None or (source.valid and value.frame_id == source.frame_id
                              and value.timestamp == source.timestamp
                              and value.valid_until <= source.valid_until)


def validate_output(value, expected, source):
    if not isinstance(value, expected):
        raise ValueError("wrong output type")
    if not current(value, source):
        raise ValueError("invalid, expired or mismatched source frame")
    if expected is DecisionTarget:
        if (value.mode not in (DecisionMode.KEEP_LANE, DecisionMode.FOLLOW,
                               DecisionMode.STOP, DecisionMode.EMERGENCY_BRAKE)
                or not number(value.target_speed) or value.target_speed < 0
                or not number(value.stop_distance) or value.stop_distance < -1
                or not isinstance(value.target_lane_id, str)):
            raise ValueError("invalid decision fields")
    elif expected is Trajectory:
        if (not number(value.target_speed) or value.target_speed < 0
                or len(value.points) < 2 or type(value.emergency_stop) is not bool
                or type(value.stop_required) is not bool
                or not number(value.stop_distance) or value.stop_distance < -1):
            raise ValueError("invalid trajectory fields")
        previous = None
        for point in value.points:
            if (not all(number(getattr(point, key)) for key in
                        ("x", "y", "speed", "heading", "relative_time"))
                    or point.speed < 0 or point.relative_time < 0):
                raise ValueError("invalid trajectory point")
            if previous is not None:
                dt = point.relative_time - previous.relative_time
                ds = math.hypot(point.x - previous.x, point.y - previous.y)
                if dt <= 0 or (ds > 1e-6 and point.speed + previous.speed <= 0):
                    raise ValueError("invalid trajectory timing")
            previous = point
        if value.emergency_stop and (value.target_speed != 0 or any(p.speed != 0 for p in value.points)):
            raise ValueError("emergency trajectory contains propulsion")
    elif expected is ControlOut:
        value.clamp()
        if getattr(source, "emergency_stop", False) and (value.throttle != 0 or value.brake <= 0):
            raise ValueError("controller ignored emergency stop")
        if value.gear == 2 and any(point.speed > 0 for point in source.points):
            raise ValueError("reverse gear with forward-only trajectory")
        if value.gear == 3 and any(point.speed > 0 for point in source.points):
            raise ValueError("parking gear with moving trajectory")
    return value
