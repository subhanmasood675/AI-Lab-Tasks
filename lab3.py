from collections import deque
from itertools import count

import heapq


# ==================================================
# GRAPH DATA
# Substitute graph supplied for this exercise.
# ==================================================

GRAPH = {

    "A": ["B", "C"],

    "B": ["A", "D", "E"],

    "C": ["F"],

    "D": ["G"],

    "E": ["G"],

    "F": ["G"],

    "G": [],

}


WEIGHTED_GRAPH = {

    "A": [("B", 2), ("C", 3), ("G", 20)],

    "B": [("A", 2), ("D", 2), ("E", 5)],

    "C": [("F", 2)],

    "D": [("G", 8)],

    "E": [("G", 2)],

    "F": [("G", 2)],

    "G": [],

}


START = "A"

GOAL = "G"


FOUND = "FOUND"

CUTOFF = "CUTOFF"

FAILURE = "FAILURE"


# ==================================================
# COMMON HELPERS
# ==================================================

def print_heading(title):

    print("\n" + "=" * 70)

    print(f"\n{title}\n")


def rebuild_path(parents, destination):

    route = []

    current = destination

    while current is not None:

        route.append(current)

        current = parents[current]

    return route[::-1]


def path_cost(graph, route):

    total = 0

    for source, destination in zip(route, route[1:]):

        edge_cost = next(

            weight

            for neighbor, weight in graph[source]

            if neighbor == destination

        )

        total += edge_cost

    return total


# ==================================================
# TASK 1: BREADTH-FIRST SEARCH
# ==================================================

def bfs_trace(graph, start, goal):

    frontier = deque([start])

    parents = {start: None}

    trace = []

    visit_order = []

    while frontier:

        current = frontier.popleft()

        visit_order.append(current)

        for neighbor in graph.get(current, []):

            if neighbor in parents:

                continue

            parents[neighbor] = current

            frontier.append(neighbor)

        trace.append((current, list(frontier)))

        if current == goal:

            route = rebuild_path(parents, goal)

            return trace, visit_order, route

    return trace, visit_order, []


# ==================================================
# TASK 2: DEPTH-FIRST SEARCH
# ==================================================

def dfs_trace(graph, start, goal):

    # Each stack entry stores a node and its ancestors.

    frontier = [(start, ())]

    trace = []

    visit_order = []

    while frontier:

        current, ancestors = frontier.pop()

        route = ancestors + (current,)

        visit_order.append(current)

        if current != goal:

            neighbors = graph.get(current, [])

            for neighbor in reversed(neighbors):

                if neighbor in route:

                    continue

                frontier.append((neighbor, route))

        frontier_snapshot = [

            list(previous_nodes + (node,))

            for node, previous_nodes in frontier

        ]

        trace.append((current, frontier_snapshot))

        if current == goal:

            return trace, visit_order, list(route)

    return trace, visit_order, []


# ==================================================
# TASK 3: ITERATIVE DEPTH-LIMITED SEARCH
# ==================================================

def dls(graph, start, goal, limit):

    if limit < 0:

        raise ValueError("Depth limit cannot be negative.")

    frontier = [(start, (start,), 0)]

    cutoff_occurred = False

    while frontier:

        current, route, depth = frontier.pop()

        if current == goal:

            return FOUND, list(route)

        available_neighbors = [

            neighbor

            for neighbor in graph.get(current, [])

            if neighbor not in route

        ]

        if depth == limit:

            cutoff_occurred = (
                cutoff_occurred or bool(available_neighbors)
            )

            continue

        for neighbor in reversed(available_neighbors):

            frontier.append(

                (
                    neighbor,

                    route + (neighbor,),

                    depth + 1,

                )

            )

    final_status = CUTOFF if cutoff_occurred else FAILURE

    return final_status, []


# ==================================================
# TASK 4: ITERATIVE DEEPENING DFS
# ==================================================

def iddfs(graph, start, goal, maximum_depth):

    attempted_limits = []

    for limit in range(1, maximum_depth + 1):

        attempted_limits.append(limit)

        status, route = dls(
            graph,
            start,
            goal,
            limit
        )

        if status != CUTOFF:

            return status, route, attempted_limits

    return CUTOFF, [], attempted_limits


# ==================================================
# TASK 5: UNIFORM-COST SEARCH
# ==================================================

def ucs(graph, start, goal):

    sequence = count()

    frontier = [

        (0, next(sequence), start)

    ]

    minimum_costs = {start: 0}

    parents = {start: None}

    processed = []

    settled = set()

    while frontier:

        current_cost, _, current = heapq.heappop(frontier)

        if current in settled:

            continue

        if current_cost != minimum_costs.get(current):

            continue

        settled.add(current)

        processed.append((current, current_cost))

        if current == goal:

            route = rebuild_path(parents, goal)

            return route, current_cost, processed

        for neighbor, edge_cost in graph.get(current, []):

            if neighbor in settled:

                continue

            candidate_cost = current_cost + edge_cost

            known_cost = minimum_costs.get(
                neighbor,
                float("inf")
            )

            if candidate_cost >= known_cost:

                continue

            minimum_costs[neighbor] = candidate_cost

            parents[neighbor] = current

            heapq.heappush(

                frontier,

                (
                    candidate_cost,

                    next(sequence),

                    neighbor,

                )

            )

    return [], float("inf"), processed


# ==================================================
# DISPLAY BFS AND DFS RESULTS
# ==================================================

def display_search_trace(label, search_function):

    print_heading(f"{label} SEARCH TRACE")

    rows, visit_order, route = search_function(
        GRAPH,
        START,
        GOAL
    )

    print("Frontier is shown after processing each node.")

    if label == "DFS":

        print("\nStack top is the rightmost entry.")

    for step, (node, frontier) in enumerate(rows, start=1):

        print(f"\nStep {step}")

        print(f"  Processed node: {node}")

        print(f"  Frontier: {frontier}")

    print(f"\nVisit order: {visit_order}")

    print(f"\nFinal path: {route}")


# ==================================================
# DISPLAY DLS AND IDDFS RESULTS
# ==================================================

def display_depth_searches():

    print_heading("DEPTH-LIMITED SEARCH")

    for limit in range(1, 5):

        status, route = dls(
            GRAPH,
            START,
            GOAL,
            limit
        )

        print(f"Depth limit: {limit}")

        print(f"Status: {status}")

        print(f"Path: {route or 'none'}")

        print("\n" + "-" * 50 + "\n")

    print_heading("ITERATIVE DEEPENING DFS")

    for maximum_depth in range(1, 5):

        status, route, attempts = iddfs(
            GRAPH,
            START,
            GOAL,
            maximum_depth
        )

        print(f"Maximum depth: {maximum_depth}")

        print(f"Status: {status}")

        print(f"Path: {route or 'none'}")

        print(f"Limits attempted: {attempts}")

        print("\n" + "-" * 50 + "\n")


# ==================================================
# DISPLAY UCS AND BFS COMPARISON
# ==================================================

def display_weighted_searches():

    print_heading("UNIFORM-COST SEARCH")

    route, cost, processed = ucs(
        WEIGHTED_GRAPH,
        START,
        GOAL
    )

    print(f"Weighted graph: {WEIGHTED_GRAPH}")

    print(f"\nProcessed node/cost order: {processed}")

    print(f"\nUCS path: {route}")

    print(f"\nReturned cost: {cost}")

    if route:

        verified_cost = path_cost(
            WEIGHTED_GRAPH,
            route
        )

        print(f"\nIndependent path-cost sum: {verified_cost}")

    else:

        print("\nIndependent path-cost sum: no route found")

    unweighted_topology = {

        node: [

            neighbor

            for neighbor, _ in connections

        ]

        for node, connections in WEIGHTED_GRAPH.items()

    }

    _, bfs_order, bfs_route = bfs_trace(
        unweighted_topology,
        START,
        GOAL
    )

    print_heading("BFS ON THE WEIGHTED GRAPH'S TOPOLOGY")

    print(f"Visit order: {bfs_order}")

    print(f"\nFewest-hop path: {bfs_route}")

    if bfs_route:

        bfs_cost = path_cost(
            WEIGHTED_GRAPH,
            bfs_route
        )

        print(f"\nBFS route cost: {bfs_cost}")

    else:

        print("\nBFS route cost: no route found")


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    print_heading("GRAPH SEARCH DEMONSTRATION")

    print(f"Start node: {START}")

    print(f"\nGoal node: {GOAL}")

    print(f"\nGraph successor order: {GRAPH}")

    search_tasks = [

        ("BFS", bfs_trace),

        ("DFS", dfs_trace),

    ]

    for label, search_function in search_tasks:

        display_search_trace(
            label,
            search_function
        )

    display_depth_searches()

    display_weighted_searches()


if __name__ == "__main__":

    main()