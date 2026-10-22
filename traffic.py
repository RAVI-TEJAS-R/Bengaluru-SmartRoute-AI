"""Optional TomTom live traffic adapter.
No synthetic traffic fallback is created.
"""

from datetime import datetime, timezone
import requests


def tomtom_edge_speeds(graph, api_key, route=None, timeout=8):
    """Fetch TomTom traffic ratios along a route."""

    if not api_key or not api_key.strip():
        return {}, "TomTom unavailable: API key is empty", None

    if not route or len(route) < 2:
        return {}, "TomTom waiting for an initial route", None

    route_edges = list(zip(route, route[1:]))
    stride = max(1, len(route_edges) // 20)
    sampled_edges = route_edges[::stride][:20]

    ratios = {}
    successes = 0
    errors = []

    url = (
        "https://api.tomtom.com/traffic/services/4/"
        "flowSegmentData/absolute/10/json"
    )

    for u, v in sampled_edges:
        try:
            node_u = graph.nodes[u]
            node_v = graph.nodes[v]

            lat = (
                float(node_u["y"]) + float(node_v["y"])
            ) / 2

            lon = (
                float(node_u["x"]) + float(node_v["x"])
            ) / 2

            response = requests.get(
                url,
                params={
                    "point": f"{lat},{lon}",
                    "key": api_key.strip(),
                },
                timeout=timeout,
            )

            if not response.ok:
                if len(errors) < 3:
                    errors.append(f"HTTP {response.status_code}")
                continue

            flow = response.json().get("flowSegmentData", {})

            current = float(flow["currentSpeed"])
            free = float(flow["freeFlowSpeed"])

            if free <= 0:
                continue

            ratio = max(
                0.12,
                min(1.0, current / free),
            )

            ratios[(str(u), str(v))] = ratio
            successes += 1

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            if len(errors) < 3:
                errors.append(type(exc).__name__)

    if successes == 0:
        detail = (
            ", ".join(errors)
            if errors
            else "No usable traffic responses"
        )

        return (
            {},
            f"TomTom unavailable on route: {detail}",
            None,
        )

    stamp = datetime.now(timezone.utc).isoformat()

    return (
        ratios,
        f"TomTom live traffic: {successes} route-edge "
        "probes succeeded (approximate edge mapping)",
        stamp,
    )


def apply_incidents(graph, report, closure_keywords):
    """Close named roads only when explicitly linked to a report."""

    g = graph.copy()
    lower = report.casefold()

    closures = [
        word
        for word in closure_keywords
        if word and word in lower
    ]

    if not closures:
        return g

    for u, v, key, data in list(
        g.edges(keys=True, data=True)
    ):
        name = str(data.get("name", "")).casefold()

        if (
            name
            and name in lower
            and any(word in lower for word in closures)
        ):
            g.remove_edge(u, v, key)

    return g


def estimate_edge_minutes(length_m, speed_kmh):
    """Estimate travel time for an edge in minutes."""

    return (
        (float(length_m) / 1000.0)
        / max(float(speed_kmh), 1.0)
        * 60.0
    )
