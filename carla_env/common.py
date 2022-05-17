import math
import shapely.geometry, shapely.affinity
import numpy as np


def pi_2_pi(theta):
    while theta > math.pi or theta < -math.pi:

        if theta > math.pi:
            theta -= 2 * math.pi

        if theta < -math.pi:
            theta += 2 * math.pi

    return theta


def cal_angle(vec_1, vec_2):
    norm_1 = np.hypot(vec_1[0], vec_1[1])
    norm_2 = np.hypot(vec_2[0], vec_2[1])
    if norm_1*norm_2 != 0:
        cos_theta = (vec_1[0]*vec_2[0] + vec_1[1]*vec_2[1])/(norm_1 * norm_2)
    else:
        print('Vector Norm is zero!')
        return None
    if cos_theta < -1 or cos_theta > 1:
        print("Angle calculation Error!")
    return math.acos(cos_theta)


def ref_waypoint(wp, max_dist = 30, dist_rate = 1.4):
    start_dist = 1
    wp_l = []
    while True:
        wp_l.append(wp.next(start_dist)[0])
        start_dist *= dist_rate
        if start_dist > max_dist:
            break
    return wp_l


def _pos(_object):
    type_obj = str(type(_object))
    if 'Actor' in type_obj or 'Vehicle' in type_obj or 'TrafficLight' in type_obj:
        return [_object.get_location().x, _object.get_location().y]
    elif 'BoundingBox' in type_obj or 'Transform' in type_obj:
        return [_object.location.x, _object.location.y]
    elif 'Vector3D' in type_obj or 'Location' in type_obj:
        return [_object.x, _object.y]
    elif 'Waypoint' in type_obj:
        return [_object.transform.location.x, _object.transform.location.y]


class RotatedRectangle(object):
    """
    This class contains method to draw rectangle and find intersection point.
    """

    def __init__(self, c_x, c_y, width, height, angle):
        self.c_x = c_x
        self.c_y = c_y
        self.w = width  # pylint: disable=invalid-name
        self.h = height  # pylint: disable=invalid-name
        self.angle = angle

    def get_contour(self):
        """
        create contour
        """
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.c_x, self.c_y)

    def intersection(self, other):
        """
        Obtain a intersection point between two contour.
        """
        return self.get_contour().intersection(other.get_contour())