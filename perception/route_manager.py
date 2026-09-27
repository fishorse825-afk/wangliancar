"""HD map lookup and lane-context normalization."""

import math
import copy
import time

from core.geometry import nearest_path_error, project_polyline
from core.interfaces import LaneContext


def _sdk_string(value):
    if value is None:
        return ""
    if hasattr(value, "GetString"):
        return value.GetString()
    return str(value)


def _vector_points(vector):
    points = []
    for index in range(int(vector.Size())):
        item = vector.GetElement(index)
        points.append((float(item.x), float(item.y), float(item.z)))
    return points


def _orient_forward(points, heading, x=None, y=None):
    if len(points) < 2:
        return points
    projection = project_polyline(points, points[0][0] if x is None else x,
                                  points[0][1] if y is None else y)
    if projection is None:
        return []
    path_heading = projection["heading"]
    if math.cos(path_heading - heading) < 0.0:
        return list(reversed(points))
    return points


class RouteManager(object):
    def __init__(self, adapter, logger, refresh_sec=0.2):
        self.adapter = adapter
        self.logger = logger
        self.refresh_sec = refresh_sec
        self.last_update = 0.0
        self.last_lane = LaneContext()

    def update(self, ego, force=False):
        if not ego.valid or not self.adapter.map_loaded:
            return LaneContext()
        now = time.monotonic()
        if not force and self.last_lane.valid and now - self.last_update < self.refresh_sec:
            lane = copy.deepcopy(self.last_lane)
            lane.lateral_offset, lane.heading_error = nearest_path_error(
                lane.center_line, ego.x, ego.y, ego.heading
            )
            return lane
        try:
            lane = self._lookup(ego)
            self.last_lane = copy.deepcopy(lane)
            self.last_update = now
            return lane
        except Exception as exc:
            self.logger.warning("车道查询失败: %s", exc)
            return LaneContext()

    def _lookup(self, ego):
        hdmap = self.adapter.hdmap
        position = hdmap.pySimPoint3D(ego.x, ego.y, ego.z)
        nearest = hdmap.getNearMostLane(position)
        if not nearest.exists:
            return LaneContext()
        lane = LaneContext()
        lane.lane_id = _sdk_string(nearest.laneId)
        sample = hdmap.getLaneSample(nearest.laneId)
        if not sample.exists:
            return lane
        points = _vector_points(sample.laneInfo.centerLine)
        if not all(math.isfinite(v) for point in points for v in point):
            return lane
        lane.center_line = _orient_forward(points, ego.heading, ego.x, ego.y)
        reversed_direction = bool(points and lane.center_line and lane.center_line[0] != points[0])
        lane.left_boundary = _vector_points(sample.laneInfo.leftBoundary)
        lane.right_boundary = _vector_points(sample.laneInfo.rightBoundary)
        for name in ("left_boundary", "right_boundary"):
            if not all(math.isfinite(v) for point in getattr(lane, name) for v in point):
                setattr(lane, name, [])
        if hasattr(hdmap, "getLaneLink"):
            link_info = hdmap.getLaneLink(nearest.laneId)
            if link_info.exists:
                lane.left_lane_id = _sdk_string(link_info.laneLink.leftNeighborLaneId)
                lane.right_lane_id = _sdk_string(link_info.laneLink.rightNeighborLaneId)
                for name, field in (("predecessor_lane_ids", "predecessorLaneIds"),
                                    ("successor_lane_ids", "successorLaneIds")):
                    vector = getattr(link_info.laneLink, field, None)
                    if vector is not None:
                        setattr(lane, name, [_sdk_string(vector.GetElement(i)) for i in range(vector.Size())])
        if hasattr(hdmap, "getLaneWidth"):
            width_info = hdmap.getLaneWidth(nearest.laneId, position)
            if width_info.exists and math.isfinite(float(width_info.width)) and float(width_info.width) > 0.1:
                lane.lane_width = float(width_info.width)
                lane.lane_width_valid = True
        if hasattr(hdmap, "getRoadMark"):
            try:
                mark_info = hdmap.getRoadMark(position, nearest.laneId)
                if mark_info.exists:
                    lane.left_mark_type = _sdk_string(mark_info.left.type)
                    lane.right_mark_type = _sdk_string(mark_info.right.type)
            except (AttributeError, TypeError, ValueError) as exc:
                self.logger.warning("车道标线查询失败: %s", exc)
        if reversed_direction:
            lane.left_boundary, lane.right_boundary = list(reversed(lane.right_boundary)), list(reversed(lane.left_boundary))
            lane.left_mark_type, lane.right_mark_type = lane.right_mark_type, lane.left_mark_type
            lane.left_lane_id, lane.right_lane_id = lane.right_lane_id, lane.left_lane_id
            lane.predecessor_lane_ids, lane.successor_lane_ids = lane.successor_lane_ids, lane.predecessor_lane_ids
        lane.lateral_offset, lane.heading_error = nearest_path_error(
            lane.center_line, ego.x, ego.y, ego.heading
        )
        lane.valid = project_polyline(lane.center_line, ego.x, ego.y) is not None
        lane.source = "hdmap"
        return lane

    def locate_target(self, target):
        """Map membership requires bounded projection and a measured lane width."""
        if not self.adapter.map_loaded:
            return "", False
        try:
            hdmap = self.adapter.hdmap
            pos = hdmap.pySimPoint3D(target.x, target.y, target.z)
            nearest = hdmap.getNearMostLane(pos)
            if not nearest.exists:
                return "", False
            sample = hdmap.getLaneSample(nearest.laneId)
            width = hdmap.getLaneWidth(nearest.laneId, pos)
            if not sample.exists or not width.exists or not math.isfinite(width.width) or width.width <= 0:
                return "", False
            points = _vector_points(sample.laneInfo.centerLine)
            if not all(math.isfinite(v) for p in points for v in p):
                return "", False
            projection = project_polyline(points, target.x, target.y)
            if (projection is None or not 0 <= projection["raw_ratio"] <= 1 or
                    projection["distance"] > width.width * 0.5):
                return "", False
            i, r = projection["index"], projection["ratio"]
            road_z = points[i][2] + r * (points[i + 1][2] - points[i][2])
            if abs(target.z - road_z) > max(2.0, target.height):
                return "", False
            return _sdk_string(nearest.laneId), True
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return "", False
