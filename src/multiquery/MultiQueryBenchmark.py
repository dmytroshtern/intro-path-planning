from IPPerfMonitor import IPPerfMonitor
import pandas as pd
from src.multiquery.MultiQueryRoundtripPlanner import MultiQueryRoundtripPlanner

import src.benchmarking.RoundtripTestSuite as ts

import matplotlib.pylab as plt
import random

def performBenchmark(totalRuns):

    visConfig = dict()
    visConfig["ntry"] = 40

    optimizations = {
        "all": (True, True),
        "none": (False, False),
        "roadmap": (True, False),
        "path": (False, True)
    }

    results = {
        "all": {},
        "none": {},
        "roadmap": {},
        "path": {}

    }

    random.seed(80085)
    seeds = random.sample(range(1, 1000000), totalRuns)

    for configName in optimizations.keys():
        print(f"@@@@@@Starting {configName} optimization")
        config = optimizations[configName]
        visConfig["optimizeRoadmap"] = config[0]
        visConfig["optimizePath"] = config[1]
        for benchmark in ts.benchList:
            print(benchmark.name)
            results[configName][benchmark.name] = {}
            for seed in seeds:
                random.seed(seed)

                planner = MultiQueryRoundtripPlanner(benchmark.collisionChecker)
                result = planner.planPath(benchmark.startList, benchmark.goalList, visConfig)

                if result["success"] == True:
                    checks = IPPerfMonitor.dataFrame().groupby(["name"]).count().loc["pointInCollision", "args"]
                    cost = result["tour_cost"]
                    IPPerfMonitor.clearData()

                    results[configName][benchmark.name]["totalChecks"] \
                        = results[configName][benchmark.name].get("totalChecks", 0) + checks

                    results[configName][benchmark.name]["totalCost"] \
                        = results[configName][benchmark.name].get("totalCost", 0) + cost

                    results[configName][benchmark.name]["totalSuccess"] \
                        = results[configName][benchmark.name].get("totalSuccess", 0) + 1

            successfulAttempts = results[configName][benchmark.name].get("totalSuccess", 0)
            if successfulAttempts > 0:
                results[configName][benchmark.name]["avgChecks"] \
                    = results[configName][benchmark.name].get("totalChecks", 0) / successfulAttempts

                results[configName][benchmark.name]["avgCost"] \
                    = results[configName][benchmark.name].get("totalCost", 0) / successfulAttempts

                results[configName][benchmark.name]["successRate"] \
                    = results[configName][benchmark.name].get("totalSuccess", 0) / totalRuns


    rows = []
    for optimization, benchmark in results.items():
        for benchmark, metrics in benchmark.items():
            row = {
                "optimization": optimization,
                "benchmark": benchmark,
                **metrics
            }
            rows.append(row)

    df = pd.DataFrame(rows)


    ######## Plot Average Cost
    pivot = df.pivot(
        index="benchmark",
        columns="optimization",
        values="avgCost"
    )

    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Average Cost")
    plt.tight_layout()
    plt.show()

    ######## Plot Average Checks
    pivot = df.pivot(
        index="benchmark",
        columns="optimization",
        values="avgChecks"
    )

    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Average Checks")
    plt.tight_layout()
    plt.show()

    ######## Plot Success Rate
    pivot = df.pivot(
        index="benchmark",
        columns="optimization",
        values="successRate"
    )

    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Success Rate")
    plt.tight_layout()
    plt.show()