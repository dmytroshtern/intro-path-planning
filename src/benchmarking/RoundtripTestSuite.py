"""Point-robot benchmarks used by the roundtrip examples."""

import math

from shapely.geometry import LineString, Point, Polygon

from notebooks.IPBenchmark import Benchmark
from notebooks.IPEnvironment import CollisionChecker


benchList = []


trap_field = {
    "obs1": LineString([(6, 18), (6, 8), (16, 8), (16, 18)]).buffer(1.0)
}
benchList.append(
    Benchmark(
        "Trap",
        CollisionChecker(trap_field),
        [[10, 15]],
        [[10.0, 1.0], [2.0, 20.0], [20.0, 20.0], [2.0, 2.0],
            [20.0, 2.0], [10.0, 20.0], [2.0, 10.0], [20.0, 10.0]],
        "Following the direct connection from goal to start leads into a trap.",
        2,
    )
)

benchList.append(
    Benchmark(
        "Trap one goal",
        CollisionChecker(trap_field),
        [[10, 15]],
        [[10.0, 1.0]],
        "Following the direct connection from goal to start leads into a trap.",
        2,
    )
)


bottleneck_field = {
    "obs1": LineString([(0, 13), (11, 13)]).buffer(0.5),
    "obs2": LineString([(13, 13), (23, 13)]).buffer(0.5),
}
benchList.append(
    Benchmark(
        "Bottleneck",
        CollisionChecker(bottleneck_field),
        [[4, 15]],
        [[18.0, 1.0], [4.0, 3.0], [18.0, 20.0], [8.0, 18.0],
            [3.0, 20.0], [20.0, 16.0], [8.0, 5.0], [15.0, 5.0]],
        "The planner has to find a narrow passage.",
        2,
    )
)


fat_bottleneck_field = {
    "obs1": Polygon([(0, 8), (11, 8), (11, 15), (0, 15)]).buffer(0.5),
    "obs2": Polygon([(13, 8), (24, 8), (24, 15), (13, 15)]).buffer(0.5),
}
benchList.append(
    Benchmark(
        "Fat bottleneck",
        CollisionChecker(fat_bottleneck_field),
        [[4, 21]],
        [[18.0, 1.0], [4.0, 3.0], [18.0, 20.0], [12.0, 4.0],
            [2.0, 20.0], [20.0, 3.0], [12.0, 18.0], [12.0, 11.0]],
        "The planner has to find a narrow passage with significant extent.",
        2,
    )
)

benchList.append(
    Benchmark(
        "Fat bottleneck one goal",
        CollisionChecker(fat_bottleneck_field),
        [[4, 21]],
        [[18.0, 1.0]],
        "The planner has to find a narrow passage with significant extent.",
        2,
    )
)


alternating_gates = {}
bar_width = 0.7
for index, x in enumerate([6, 11, 16, 21, 26]):
    bottom, top = (0, 9) if index % 2 == 0 else (5, 14)
    alternating_gates[f"bar{index}"] = Polygon(
        [
            (x - bar_width / 2, bottom),
            (x + bar_width / 2, bottom),
            (x + bar_width / 2, top),
            (x - bar_width / 2, top),
        ]
    )

benchList.append(
    Benchmark(
        "Alternating Gates",
        CollisionChecker(alternating_gates, limits=[[0, 30], [0, 14]]),
        [[2, 7]],
        [[28, 7]],
        "Alternating walls force a zig-zag path through narrow gates.",
        2,
    )
)


my_field = {
    "L": Polygon([(10, 16), (10, 11), (13, 11), (13, 12), (11, 12), (11, 16)]),
    "T": Polygon(
        [(14, 16), (14, 15), (15, 15), (15, 11), (16, 11),
         (16, 15), (17, 15), (17, 16)]
    ),
    "C": Polygon(
        [(19, 16), (19, 11), (22, 11), (22, 12),
         (20, 12), (20, 15), (22, 15), (22, 16)]
    ),
    "Antenna_L": Polygon([(3, 12), (1, 16), (2, 16), (4, 12)]),
    "Antenna_Head_L": Point(1.5, 16).buffer(1),
    "Antenna_R": Polygon([(7, 12), (9, 16), (8, 16), (6, 12)]),
    "Antenna_Head_R": Point(8.5, 16).buffer(1),
    "Rob_Head": Polygon([(2, 13), (2, 8), (8, 8), (8, 13)]),
}
benchList.append(
    Benchmark(
        "MyField",
        CollisionChecker(my_field),
        [[4, 21]],
        [[18, 1], [5, 5], [14, 14], [21, 1]],
        "The planner has to pass a robot head and the letters LTC.",
        2,
    )
)


def _star_point(angle, radius, center):
    radians = math.radians(angle)
    return [
        radius * math.cos(radians) + center[0],
        radius * math.sin(radians) + center[1],
    ]


def _star_polygon_and_goals(inner_radius, outer_radius, center, tips):
    polygon_points = []
    goals = []
    angle = 90
    angle_step = 360 / (tips * 2)

    for _ in range(tips):
        polygon_points.append(_star_point(angle, outer_radius, center))
        angle += angle_step
        polygon_points.append(_star_point(angle, inner_radius, center))
        goals.append(_star_point(angle, inner_radius + 0.5, center))
        angle += angle_step

    return Polygon(polygon_points), goals


star_polygon, star_goals = _star_polygon_and_goals(3, 9, [10, 10], 7)
benchList.append(
    Benchmark(
        "Star",
        CollisionChecker({"star": star_polygon}),
        [[0, 0]],
        star_goals,
        "Star with goals between its rays.",
        2,
    )
)

balls = dict()
size = 1.25
numBalls = 25
upperBound = 20
goalList = [[10, 10], [5, 20], [15.329290265318571, 4.447833402385777], [16.583613504242788, 18.359272325420363], [13.503408990620507, 4.232410865747257], [2.5, 10], [7, 5.3534655251254], [16.75, 12.25]]

ballCenters = [[1.9949083781679422, 2.656546872930014], [8.737593509737568, 3.0463044638303423], [18.1227786810822, 1.7249196559441256], [11.187838760644611, 7.311756271683917], [17.189645759658003, 4.033585537717428], [7.4071164850762345, 7.342131497950151], [5.658103697834611, 1.0317405180052344], [9.289240074889149, 2.6646564952580345], [12.35438982527182, 2.3276622677244663], [6.993720500981403, 9.518112457831931], [18.202574393405477, 2.7625266661914067], [3.7013262241985205, 16.011365042817232], [1.4034835215227957, 18.311860413716094], [11.894753627937616, 6.041112399969703], [16.91891523970755, 15.607100202119687], [7.525781004963327, 16.228435234548627], [4.930386942656117, 12.569147997620513], [11.014424091066916, 16.377431968232294], [6.989999892103774, 8.566524877890085], [15.616302708364552, 10.742290543417235], [14.011907154559115, 13.128712021804477], [13.666494990390438, 17.911872840201145], [10.236592932727715, 13.700934216159794], [8.115911004627183, 1.6049848280809353]]

for i, center in enumerate(ballCenters):
    balls[f"{i}"] = Point(center[0], center[1]).buffer(size)
    

description = "Pseudorandom balls blocking the way"
benchList.append(Benchmark("Balls", CollisionChecker(balls), [[0, 0]], goalList, description, 2))
