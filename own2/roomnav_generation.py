ISLAND_RADIUS_LIMIT = 1.5
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal, RoomGoal
from numpy import float64

from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union
import numpy as np
from shapely.geometry import Polygon, Point

from shapely.ops import nearest_points
from shapely.geometry import Polygon, Point
import math

def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.

    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(
    s: Sequence[float],
    t: Sequence[float],
    sim: "HabitatSim",
    near_dist: float,
    far_dist: float,
    geodesic_to_euclid_ratio: float,
) -> Union[Tuple[bool, float], Tuple[bool, int]]:
    euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)
    if np.abs(s[1] - t[1]) > 0.5:  # check height difference to assure s and
        #  t are from same floor
        return False, 0
    d_separation = sim.geodesic_distance(s, [t])
    if d_separation == np.inf:
        return False, 0
    if not near_dist <= d_separation <= far_dist:
        return False, 0
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return False, 0
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 0
    return True, d_separation

def _create_episode(
    episode_id: Union[int, str],
    scene_id: str,
    start_position: List[float],
    start_rotation: List[Union[int, float64]],
    target_point,
    target_points,
    room_bounds,
    semantic_id,
    info: Optional[Dict[str, float]] = None,
) -> Optional[NavigationEpisode]:
    goals = [RoomGoal(position=target_point,semantic_id=semantic_id, rooms_bound_points=target_points, rooms_bounds=room_bounds)]
    return NavigationEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        info=info,
    )



def generate_roomnav_episode(
    scene,
    sim: "HabitatSim",
    annotation_json: dict,
    num_episodes: int = -1,
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 10,
) -> Generator[NavigationEpisode, None, None]:
    episode_count = 0

    while episode_count < num_episodes or num_episodes < 0:
                print(episode_count)
                source_position = sim.sample_navigable_point()
                semantics = ["storage", "meeting room", "office space", "office", "elevators", "toilet", "stairs", "hallway"]
                #semantic_id = np.random.randint(len(semantics))
                semantic_id = 5
                # semantics = ["storage", "meeting room", "office space",
                #              "office", "café", "elevators", "server", "toilet",
                #              "stairs", "hallway"]
                rooms = []
                for room in annotation_json["regions"]:
                    if room["semantics"] == semantics[semantic_id]:
                        rooms.append(room)
                rooms_bound_points = []
                rooms_bounds = []
                is_compatible = False
                for room in rooms:
                    rooms_bounds.append(room["points"])

                    poly = Polygon(rooms_bounds[-1])
                    minx, miny, maxx, maxy = poly.bounds
                    for _retry in range(20):
                        point = Point([np.random.uniform(minx, maxx),np.random.uniform(miny, maxy)])
                        if not point.within(poly):
                            continue
                        tmp = [point.coords[0][0],0.2, point.coords[0][1]]
                        if not sim.is_navigable(tmp):
                            continue

                        is_compatible, dist = is_compatible_episode(
                            source_position,
                            tmp,
                            sim,
                            near_dist=closest_dist_limit,
                            far_dist=furthest_dist_limit,
                            geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
                        )
                        if is_compatible:
                            break

                    tmp_point = Point([source_position[0],source_position[2]])
                    if tmp_point.within(poly):
                        is_compatible = False
                    angle = np.random.uniform(0, 2 * np.pi)
                    source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

                    poly_n = poly.buffer(-0.5)
                    points = []
                    for i in range(-1, len(poly.exterior.coords.xy[0]) - 1):
                        p1 = [poly_n.exterior.coords.xy[0][i],
                              poly_n.exterior.coords.xy[1][i]]
                        p2 = [poly_n.exterior.coords.xy[0][i + 1],
                              poly_n.exterior.coords.xy[1][i + 1]]
                        if p1 != p2:
                            v = [p2[0] - p1[0], p2[1] - p1[1]]
                            v2 = math.sqrt(
                                math.pow(v[0], 2) + math.pow(v[1], 2))
                            v3 = [v[0] / v2, v[1] / v2]

                            distances = []
                            d = 0
                            while True:
                                d += 0.5
                                if d > v2:
                                    break
                                distances.append(d)
                            for i in distances:
                                vt = [v3[0] * i, v3[1] * i]
                                points.append([p1[0] + vt[0],0.2, p1[1] + vt[1]])
                    point = poly.centroid
                    new_point = [point.x, 0.2, point.y]
                    if points == []:
                        points.append(new_point)
                    rooms_bound_points.append(points)
                if is_compatible:
                    if scene == 'beacon-6':
                        episode = _create_episode(
                            episode_id=episode_count,
                            scene_id="beacon/beacon-6.glb",
                            start_position=source_position,
                            start_rotation=source_rotation,
                            target_point=new_point,
                            target_points=rooms_bound_points,
                            semantic_id=semantic_id,
                            room_bounds=rooms_bounds,
                            info={"geodesic_distance": dist}
                        )
                    elif scene == 'beacon-7':
                        episode = _create_episode(
                            episode_id=episode_count,
                            scene_id="beacon/beacon-7-untrimmed.glb",
                            start_position=source_position,
                            start_rotation=source_rotation,
                            target_point=new_point,
                            target_points=rooms_bound_points,
                            semantic_id=semantic_id,
                            room_bounds=rooms_bounds,
                            info={"geodesic_distance": dist}
                        )

                    episode_count += 1
                    yield episode
