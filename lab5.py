import heapq


# ==================================================
# GRAPH AND HEURISTIC DATA
# ==================================================

GRAPH = {

    "S": [("A", 2), ("B", 4)],

    "A": [("C", 1)],

    "B": [("D", 1), ("G", 4)],

    "C": [("G", 3)],

    "D": [("G", 2)],

    "G": [],

}


HEURISTIC = {

    "S": 5,

    "A": 3,

    "B": 2,

    "C": 2,

    "D": 1,

    "G": 0,

}


START = "S"

GOAL = "G"


# ==================================================
# COMMON HELPERS
# ==================================================

def show_heading(title):

    print("\n" + "=" * 70)

    print(f"\n{title}\n")


def reconstruct_path(parents, destination):

    route = []

    current = destination

    while current is not None:

        route.append(current)

        current = parents[current]

    return route[::-1]


def path_cost(graph, route):

    if not route:

        return float("inf")

    total = 0

    for source, target in zip(route, route[1:]):

        edge_cost = next(

            weight

            for neighbor, weight in graph.get(source, [])

            if neighbor == target

        )

        total += edge_cost

    return total


# ==================================================
# SHARED A* / UCS SEARCH ENGINE
# Priority order: (f, g, node)
# ==================================================

def run_search(graph, heuristic, start, goal):

    frontier = [

        (heuristic[start], 0, start)

    ]

    best_costs = {start: 0}

    parents = {start: None}

    processed = []

    expanded = []

    stale_entries = []

    expansion_steps = []

    final_path = []

    final_cost = float("inf")

    while frontier:

        priority, current_cost, current = heapq.heappop(frontier)

        # Skip entries that have been replaced by a cheaper route.

        if current_cost > best_costs.get(current, float("inf")):

            stale_entries.append(

                (current, current_cost, priority)

            )

            continue

        processed.append(

            (
                current,

                current_cost,

                heuristic[current],

                priority,

            )

        )

        if current == goal:

            final_path = reconstruct_path(
                parents,
                goal
            )

            final_cost = current_cost

            break

        expanded.append(current)

        generated = []

        for neighbor, edge_cost in graph.get(current, []):

            candidate_cost = current_cost + edge_cost

            previous_cost = best_costs.get(
                neighbor,
                float("inf")
            )

            if candidate_cost >= previous_cost:

                continue

            best_costs[neighbor] = candidate_cost

            parents[neighbor] = current

            candidate_priority = (
                candidate_cost + heuristic[neighbor]
            )

            heapq.heappush(

                frontier,

                (
                    candidate_priority,

                    candidate_cost,

                    neighbor,

                )

            )

            generated.append(

                {
                    "source": current,

                    "target": neighbor,

                    "edge_cost": edge_cost,

                    "g": candidate_cost,

                    "h": heuristic[neighbor],

                    "f": candidate_priority,

                }

            )

        expansion_steps.append(

            {
                "node": current,

                "g": current_cost,

                "h": heuristic[current],

                "f": priority,

                "generated": generated,

                "frontier": sorted(frontier),

            }

        )

    return {

        "path": final_path,

        "cost": final_cost,

        "processed": processed,

        "expanded": expanded,

        "stale_skipped": stale_entries,

        "steps": expansion_steps,

    }


# ==================================================
# DISPLAY EXPANSION TRACE
# ==================================================

def display_first_expansions(result, heuristic, limit=4):

    available_steps = result["steps"][:limit]

    show_heading(
        f"FIRST {len(available_steps)} A* EXPANSIONS "
        f"(UP TO {limit})"
    )

    for number, step in enumerate(available_steps, start=1):

        print(f"Expansion {number}: select {step['node']}")

        print(

            f"\n  g={step['g']}; "

            f"h={step['h']}; "

            f"f=g+h={step['g']}+{step['h']}={step['f']}"

        )

        print("\n  Accepted successor updates:")

        if not step["generated"]:

            print("\n    No improved successor routes.")

        for successor in step["generated"]:

            print(

                f"\n    {successor['source']}"

                f"->{successor['target']}: "

                f"g={step['g']}+{successor['edge_cost']}"

                f"={successor['g']}; "

                f"h={successor['h']}; "

                f"f={successor['g']}+{successor['h']}"

                f"={successor['f']}"

            )

        print("\n  Frontier entries (node: g, h, f):")

        for priority, cost, node in step["frontier"]:

            print(

                f"\n    {node}: "

                f"g={cost}, "

                f"h={heuristic[node]}, "

                f"f={priority}"

            )

        if step["frontier"]:

            priority, cost, node = step["frontier"][0]

            print(

                f"\n  Lowest queue entry: {node}; "

                f"priority tuple "

                f"(f={priority}, g={cost}, node={node})"

            )

        print("\n" + "-" * 60 + "\n")


# ==================================================
# DISPLAY SEARCH RESULTS
# ==================================================

def display_result(label, graph, result):

    show_heading(label)

    verified_cost = path_cost(
        graph,
        result["path"]
    )

    route_text = (
        " -> ".join(result["path"])
        if result["path"]
        else "No path found"
    )

    details = [

        ("Processed (node, g, h, f)", result["processed"]),

        ("Expanded (goal excluded)", result["expanded"]),

        ("Expansion count", len(result["expanded"])),

        ("Returned path", route_text),

        ("Returned cost", result["cost"]),

        ("Independent path-cost sum", verified_cost),

        ("Stale entries skipped", result["stale_skipped"]),

    ]

    for field, value in details:

        print(f"{field}: {value}\n")

    assert result["cost"] == verified_cost


# ==================================================
# CALCULATE TRUE REMAINING COSTS
# Reverse graph search starts from the goal.
# ==================================================

def calculate_remaining_costs(graph, goal):

    reverse_graph = {goal: []}

    for source, edges in graph.items():

        reverse_graph.setdefault(source, [])

        for target, weight in edges:

            reverse_graph.setdefault(target, []).append(

                (source, weight)

            )

    distances = {

        node: float("inf")

        for node in reverse_graph

    }

    distances[goal] = 0

    frontier = [(0, goal)]

    while frontier:

        distance, current = heapq.heappop(frontier)

        if distance > distances[current]:

            continue

        for predecessor, weight in reverse_graph[current]:

            candidate_distance = distance + weight

            if candidate_distance >= distances[predecessor]:

                continue

            distances[predecessor] = candidate_distance

            heapq.heappush(

                frontier,

                (candidate_distance, predecessor)

            )

    return distances


# ==================================================
# HEURISTIC AUDIT
# ==================================================

def display_heuristic_audit(
    graph,
    heuristic,
    goal,
    title="HEURISTIC AUDIT"
):

    show_heading(title)

    remaining_costs = calculate_remaining_costs(
        graph,
        goal
    )

    all_admissible = True

    all_consistent = True

    print("Admissibility checks:\n")

    for node in remaining_costs:

        estimate = heuristic[node]

        actual_cost = remaining_costs[node]

        admissible = estimate <= actual_cost

        all_admissible = all_admissible and admissible

        print(

            f"  {node}: "

            f"h={estimate}; "

            f"true remaining cost={actual_cost}; "

            f"admissible={admissible}\n"

        )

    print("\nConsistency edge checks:\n")

    for source, edges in graph.items():

        for target, weight in edges:

            left = heuristic[source]

            right = weight + heuristic[target]

            consistent = left <= right

            all_consistent = all_consistent and consistent

            print(

                f"  {source}->{target}: "

                f"{left} <= {weight}+{heuristic[target]}"

                f"={right}: {consistent}\n"

            )

    goal_is_zero = heuristic[goal] == 0

    print(f"All admissibility checks passed: {all_admissible}\n")

    print(f"All edge consistency checks passed: {all_consistent}\n")

    print(f"Goal heuristic is zero: {goal_is_zero}\n")

    return remaining_costs


# ==================================================
# EXPERIMENT COMPARISON
# ==================================================

def display_comparison(experiments):

    show_heading("FINAL EXPERIMENT COMPARISON")

    for name, result in experiments:

        route = (
            " -> ".join(result["path"])
            if result["path"]
            else "No path found"
        )

        print(f"Experiment: {name}")

        print(f"\n  Route: {route}")

        print(f"\n  Cost: {result['cost']}")

        print(f"\n  Expanded nodes: {result['expanded']}")

        print(
            f"\n  Expansion count: "
            f"{len(result['expanded'])}"
        )

        print("\n" + "-" * 60 + "\n")


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    show_heading("GRAPH SEARCH EXPERIMENTS")

    print(f"Start node: {START}\n")

    print(f"Goal node: {GOAL}\n")

    print(f"Graph: {GRAPH}\n")

    print(f"Baseline heuristic: {HEURISTIC}\n")

    # ----------------------------------------------
    # Experiment 1: Baseline A*
    # ----------------------------------------------

    baseline_result = run_search(
        GRAPH,
        HEURISTIC,
        START,
        GOAL
    )

    display_first_expansions(
        baseline_result,
        HEURISTIC,
        limit=4
    )

    display_result(
        "EXPERIMENT 1: BASELINE A*",
        GRAPH,
        baseline_result
    )

    # ----------------------------------------------
    # Experiment 2: UCS
    # ----------------------------------------------

    zero_heuristic = {

        node: 0

        for node in HEURISTIC

    }

    ucs_result = run_search(
        GRAPH,
        zero_heuristic,
        START,
        GOAL
    )

    display_result(
        "EXPERIMENT 2: UCS WITH ZERO HEURISTIC",
        GRAPH,
        ucs_result
    )

    # ----------------------------------------------
    # Experiment 3: Half-sized heuristic values
    # ----------------------------------------------

    reduced_heuristic = {

        node: estimate / 2

        for node, estimate in HEURISTIC.items()

    }

    reduced_result = run_search(
        GRAPH,
        reduced_heuristic,
        START,
        GOAL
    )

    display_result(
        "EXPERIMENT 3: A* WITH HALF-SIZED HEURISTICS",
        GRAPH,
        reduced_result
    )

    # ----------------------------------------------
    # Experiment 4: Overestimate h(C)
    # ----------------------------------------------

    overestimated_heuristic = {

        **HEURISTIC,

        "C": 12,

    }