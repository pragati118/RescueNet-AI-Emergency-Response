import heapq
import math

ROAD_GRAPH = {
    "A1": {"B2": 2.0, "C3": 3.5},
    "B2": {"A1": 2.0, "C3": 1.5, "D4": 2.5},
    "C3": {"A1": 3.5, "B2": 1.5, "E5": 2.0},
    "D4": {"B2": 2.5, "E5": 1.0, "F6": 3.0},
    "E5": {"C3": 2.0, "D4": 1.0, "G7": 2.5},
    "F6": {"D4": 3.0, "G7": 1.5},
    "G7": {"E5": 2.5, "F6": 1.5, "EMERGENCY": 1.0},
    "EMERGENCY": {"G7": 1.0}
}

NODE_COORDINATES = {
    "A1": (0, 0),
    "B2": (2, 1),
    "C3": (4, 0),
    "D4": (3, 3),
    "E5": (5, 2),
    "F6": (6, 4),
    "G7": (7, 3),
    "EMERGENCY": (8, 4)
}

def heuristic(node, goal):
    x1, y1 = NODE_COORDINATES[node]
    x2, y2 = NODE_COORDINATES[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def astar_route(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()

            total_distance = g_score[goal]
            estimated_time = round(total_distance * 2.5, 1)

            return {
                "path": path,
                "distance": round(total_distance, 2),
                "time_minutes": estimated_time
            }

        for neighbor, cost in ROAD_GRAPH[current].items():
            new_cost = g_score[current] + cost

            if neighbor not in g_score or new_cost < g_score[neighbor]:
                g_score[neighbor] = new_cost
                f_score = new_cost + heuristic(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(open_set, (f_score, neighbor))

    return None