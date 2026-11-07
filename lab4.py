import heapq


# ==================================================
# GRAPH AND HEURISTIC DATA
# ==================================================

GRAPH = {

    "S": [("A", 1), ("B", 4)],

    "A": [("C", 2)],

    "B": [("G", 5)],

    "C": [("G", 3)],

    "G": [],

}


HEURISTIC = {

    "S": 5,

    "A": 4,

    "B": 4,

    "C": 2,

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

        weight = next(

            cost

            for neighbor, cost in graph.get(source, [])

            if neighbor == target

        )

        total += weight

    return total


# ==================================================
# SHARED SEARCH ENGINE
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
# FIRST THREE EXPANSIONS
# ==================================================

def display_first_expansions(result, heuristic, limit=3):

    show_heading(f"FIRST {limit} A* EXPANSIONS")

    for number, step in enumerate(
        result["steps"][:limit],
        start=1
    ):

        print(f"Expansion {number}: select {step['node']}")

        print(

            f"\n  g={step['g']}; "

            f"h={step['h']}; "

            f"f=g+h={step['g']}+{step['h']}={step['f']}"

        )

        print("\n  Generate successors in listed order:")

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

        print("\n  Remaining frontier (node: g, h, f):")

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

                f"\n  Next queue entry: {node}; "

                f"priority tuple "

                f"(f={priority}, g={cost}, node={node})"

            )

        print("\n" + "-" * 60 + "\n")


# ==================================================
# SEARCH RESULT DISPLAY
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

        ("Processed (goal included)", result["processed"]),

        ("Expanded (goal excluded)", result["expanded"]),

        ("Expansion count", len(result["expanded"])),

        ("Returned path", route_text),

        ("Returned cost", result["cost"]),

        ("Independent path-cost sum", verified_cost),

        ("Stale entries skipped", result["stale_skipped"]),

    ]

    for label, value in details:

        print(f"{label}: {value}\n")

    assert result["cost"] == verified_cost


# ==================================================
# CALCULATE TRUE REMAINING COSTS
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

            candidate = distance + weight

            if candidate >= distances[predecessor]:

                continue

            distances[predecessor] = candidate

            heapq.heappush(

                frontier,

                (candidate, predecessor)

            )

    return distances


# ==================================================
# HEURISTIC AUDIT
# ==================================================

def display_heuristic_audit(graph, heuristic, goal):

    show_heading("HEURISTIC AUDIT")

    remaining_costs = calculate_remaining_costs(
        graph,
        goal
    )

    print("Admissibility checks:\n")

    for node in graph:

        estimate = heuristic[node]

        actual = remaining_costs[node]

        print(

            f"  {node}: "

            f"supplied h={estimate}; "

            f"true remaining cost={actual}; "

            f"admissible={estimate <= actual}\n"

        )

    print("\nConsistency edge checks:\n")

    for source, edges in graph.items():

        for target, weight in edges:

            left = heuristic[source]

            right = weight + heuristic[target]

            print(

                f"  {source}->{target}: "

                f"{left} <= {weight}+{heuristic[target]}"

                f"={right}: {left <= right}\n"

            )

    return remaining_costs


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    astar_result = run_search(
        GRAPH,
        HEURISTIC,
        START,
        GOAL
    )

    display_first_expansions(
        astar_result,
        HEURISTIC
    )

    display_result(
        "A* WITH SUPPLIED HEURISTIC",
        GRAPH,
        astar_result
    )

    zero_heuristic = dict.fromkeys(
        HEURISTIC,
        0
    )

    ucs_result = run_search(
        GRAPH,
        zero_heuristic,
        START,
        GOAL
    )

    display_result(
        "UCS CONTROL: ZERO HEURISTIC",
        GRAPH,
        ucs_result
    )

    modified_heuristic = {

        **HEURISTIC,

        "C": 10,

    }

    modified_result = run_search(
        GRAPH,
        modified_heuristic,
        START,
        GOAL
    )

    display_result(
        "A* OVERESTIMATE EXPERIMENT: ONLY h(C)=10",
        GRAPH,
        modified_result
    )

    remaining_costs = display_heuristic_audit(
        GRAPH,
        HEURISTIC,
        GOAL
    )

    show_heading("OVERESTIMATE CHECK")

    print(

        f"h(C)={modified_heuristic['C']} > "

        f"true remaining cost={remaining_costs['C']}"

    )

    edge_weight = next(

        weight

        for neighbor, weight in GRAPH["C"]

        if neighbor == GOAL

    )

    right_side = (
        edge_weight + modified_heuristic[GOAL]
    )

    print(

        f"\nOn C->{GOAL}: "

        f"{modified_heuristic['C']} <= "

        f"{edge_weight}+{modified_heuristic[GOAL]} "

        f"is {modified_heuristic['C'] <= right_side}"

    )


if __name__ == "__main__":

    main()