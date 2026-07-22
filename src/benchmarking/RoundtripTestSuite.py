# coding: utf-8

from IPBenchmark import Benchmark 
from IPEnvironment import CollisionChecker
from shapely.geometry import Point, Polygon, LineString
import shapely.affinity
import math
import numpy as np


benchList = list()


# -----------------------------------------
trapField = dict()
trapField["obs1"] =   LineString([(6, 18), (6, 8), (16, 8), (16,18)]).buffer(1.0)
description = "Following the direct connection from goal to start would lead the algorithm into a trap."
benchList.append(Benchmark("Trap", CollisionChecker(trapField), [[10,15]], [[10,1]], description, 2))

# -----------------------------------------
bottleNeckField = dict()
bottleNeckField["obs1"] = LineString([(0, 13), (11, 13)]).buffer(.5)
bottleNeckField["obs2"] = LineString([(13, 13), (23,13)]).buffer(.5)
description = "Planer has to find a narrow passage."
benchList.append(Benchmark("Bottleneck", CollisionChecker(bottleNeckField), [[4,15]], [[18,1]], description, 2))

# -----------------------------------------
fatBottleNeckField = dict()
fatBottleNeckField["obs1"] = Polygon([(0, 8), (11, 8),(11, 15), (0, 15)]).buffer(.5)
fatBottleNeckField["obs2"] = Polygon([(13, 8), (24, 8),(24, 15), (13, 15)]).buffer(.5)
description = "Planer has to find a narrow passage with a significant extend."
benchList.append(Benchmark("Fat bottleneck", CollisionChecker(fatBottleNeckField), [[4,21]], [[18,1]], description, 2))

# ----------------------------------------- 2 additional benchmarks

alternatingGates = dict()

limits = [[0, 30], [0, 14]]
bar_width = 0.7

for idx, x in enumerate([6, 11, 16, 21, 26]):
    if idx % 2 == 0:
        # obstacle grows from bottom, gap is at the top
        alternatingGates["bar" + str(idx)] = Polygon([
            (x - bar_width / 2, 0),
            (x + bar_width / 2, 0),
            (x + bar_width / 2, 9),
            (x - bar_width / 2, 9)
        ])
    else:
        # obstacle grows from top, gap is at the bottom
        alternatingGates["bar" + str(idx)] = Polygon([
            (x - bar_width / 2, 5),
            (x + bar_width / 2, 5),
            (x + bar_width / 2, 14),
            (x - bar_width / 2, 14)
        ])

start = [[2, 7]]
goal = [[28, 7]]

description = ("The direct path from start to goal is blocked by alternating walls. The planner has to find a zig-zag path through several narrow gates.")
benchList.append(
    Benchmark(
        "Alternating Gates",
        CollisionChecker(alternatingGates, limits=limits),
        start,
        goal,
        description,
        2
    )
)

# -----------------------------------------

escapeChamber = dict()

cx, cy = 11.0, 11.0
radius = 6.0

# Create an almost complete circular wall.
# The missing part is the opening/gap.
angles = np.linspace(np.deg2rad(110), np.deg2rad(430), 120)

wall_points = [
    (cx + np.cos(a) * radius, cy + np.sin(a) * radius)
    for a in angles
]

escapeChamber["almost_ring"] = LineString(wall_points).buffer(0.45)

start = [[11, 11]]
goal = [[20, 11]]

description = (
    "The start configuration is inside an almost closed chamber. "
    "The goal is outside the chamber, so the planner must find the small opening instead of trying the direct path."
)

benchList.append(
    Benchmark(
        "Escape Chamber",
        CollisionChecker(escapeChamber),
        start,
        goal,
        description,
        2
    )
)

# -----------------------------------------

myField = dict()
myField["L"] = Polygon([(10, 16), (10, 11), (13, 11), (13,12), (11,12), (11,16)])
myField["T"] = Polygon([(14,16), (14, 15), (15, 15),(15,11), (16,11), (16,15), (17, 15), (17, 16)])
myField["C"] = Polygon([(19, 16), (19, 11), (22, 11), (22, 12), (20, 12), (20, 15), (22, 15), (22, 16)])

myField["Antenna_L"] = Polygon([(3, 12), (1, 16), (2, 16), (4, 12)])
myField["Antenna_Head_L"] = Point(1.5, 16).buffer(1)

myField["Antenna_R"] = Polygon([(7, 12), (9, 16), (8, 16), (6, 12)])
myField["Antenna_Head_R"] = Point(8.5, 16).buffer(1)

myField["Rob_Head"] = Polygon([(2, 13), (2, 8), (8, 8), (8, 13)])
description = "Planer has to find a passage past a robot head and the print of the LTC."
benchList.append(Benchmark("MyField", CollisionChecker(myField), [[4,21]], [[18,1], [5, 5], [14, 14], [21, 1]], description, 2))

# -----------------------------------------
def calcStarPoint(angleDeg, radius, starCenter):
    x = radius * math.cos(math.radians(angleDeg)) + starCenter[0]
    y = radius * math.sin(math.radians(angleDeg)) + starCenter[1]
    return [x, y]

def calcStarPolygonAndGoals(innerRad, outerRad, starCenter, tips):
    starEndpoints = []
    goalList = []
    currentAngleDeg = 90
    angleShift = 360/(tips*2)
    for i in range(tips):
        starEndpoints.append(calcStarPoint(currentAngleDeg, outerRad, starCenter))
        currentAngleDeg += angleShift
        starEndpoints.append(calcStarPoint(currentAngleDeg, innerRad, starCenter))
        goalList.append(calcStarPoint(currentAngleDeg, (innerRad + 0.5), starCenter))
        currentAngleDeg += angleShift
    return Polygon(starEndpoints), goalList

star = dict()
innerRad = 1
outerRad = 8
starCenter = [10,10]
tips = 7

starEndpoints, goalList = calcStarPolygonAndGoals(3, 9, starCenter, tips)
star["star"] = starEndpoints
description = "Star with goals between the stars rays"
benchList.append(Benchmark("Star", CollisionChecker(star), [[0, 0]], goalList, description, 2))

