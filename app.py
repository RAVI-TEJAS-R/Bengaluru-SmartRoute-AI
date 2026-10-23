import os
import time

import streamlit as st
import pandas as pd
import networkx as nx
import folium
import osmnx as ox

from streamlit_folium import st_folium
from algos import (
    binary_search,
    dijkstra,
    dijkstra_with_stats,
    astar_search,
    bidirectional_dijkstra,
    horspool_search,
    RSABox,
    route_geometry,
)
from traffic import (
    tomtom_edge_speeds,
    apply_incidents,
    estimate_edge_minutes,
)
def get_tomtom_key():
    key = os.getenv("TOMTOM_API_KEY", "")
    if key:
        return key

    try:
        return st.secrets.get("TOMTOM_API_KEY", "")
    except Exception:
        return ""


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bengaluru SmartRoute AI",
    page_icon="🚦",
    layout="wide",
)
st.warning("DEPLOY CHECK 2026-10-23-A")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.3rem;
        max-width: 1500px;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #101827, #163b50);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        color: white;
    }
    .hero p {
        color: #cbd5e1;
        margin: .4rem 0 0;
        line-height: 1.55;
    }
    [data-testid="stMetric"] {
        background: rgba(120,140,160,.08);
        padding: 14px;
        border-radius: 12px;
        min-height: 105px;
    }
    div.stFormSubmitButton > button[kind="primary"] {
        min-height: 2.8rem;
        font-weight: 650;
        border-radius: 10px;
    }
    div[data-testid="stExpander"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🚦 Bengaluru SmartRoute AI</h1>
        <p>
        Find a route across Bengaluru, check available traffic data,
        and explore how the routing system works.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD THE OPENSTREETMAP ROAD GRAPH
# --------------------------------------------------


GRAPH_PATH = "data/graph.graphml"
GRAPH_DRIVE_FILE_ID = "1bsBRgLb0REv3e1iS4tKagpifyS9a4Fea"


def ensure_graph_file():
    """Download the road graph when it is not already available."""
    if (
        os.path.exists(GRAPH_PATH)
        and os.path.getsize(GRAPH_PATH) > 1_000_000
    ):
        return True

    try:
        import gdown

        os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)

        with st.spinner(
            "Downloading the Bengaluru road network on first launch..."
        ):
            downloaded = gdown.download(
            id=GRAPH_DRIVE_FILE_ID,
            output=GRAPH_PATH,
            quiet=False
            )

        if (
            not downloaded
            or not os.path.exists(GRAPH_PATH)
            or os.path.getsize(GRAPH_PATH) < 1_000_000
        ):
            st.error(
                "Road graph download failed. Check that the Google Drive "
                "file is shared with Anyone with the link as Viewer."
            )
            return False

        return True

    except Exception as error:
        st.error(f"Road graph download failed: {error}")
        return False


@st.cache_resource
def load_graph():
    if not ensure_graph_file():
        return None

    try:
        # OSMnx restores the road graph and its numeric
        # geographic attributes from the GraphML file.
        graph = ox.load_graphml(filepath=GRAPH_PATH)

        # Ensure x and y coordinates are numeric.
        for node_id, data in graph.nodes(data=True):
            if "x" in data:
                data["x"] = float(data["x"])
            if "y" in data:
                data["y"] = float(data["y"])

        return graph

    except Exception as error:
        st.error(f"Could not load the road graph: {error}")
        return None



G = load_graph()

if G is None:
    st.error("The Bengaluru road network could not be loaded.")
    st.code("python build_graph.py")
    st.stop()


# --------------------------------------------------
# CREATE SEARCHABLE ROAD AND JUNCTION LABELS
# --------------------------------------------------

def node_label(item):
    node_id, data = item

    name = str(data.get("name", "")).split(",")[0].strip()

    if name:
        return f"{name} · {node_id}"

    return f"Junction {node_id}"


@st.cache_resource
def build_label_index():
    """Build the large road-label index once, not on every Streamlit rerun."""
    mapping = {
        node_label((node_id, data)): node_id
        for node_id, data in G.nodes(data=True)
    }
    sorted_labels = sorted(mapping.keys(), key=str.casefold)
    lower_labels = [label.casefold() for label in sorted_labels]
    return mapping, sorted_labels, lower_labels


label_to_node, labels, labels_lower = build_label_index()


# --------------------------------------------------
# GEOGRAPHIC LOCATION LOOKUP
# --------------------------------------------------


@st.cache_data(ttl=86400)
def geocode_location(query):
    """Resolve common Bengaluru locations locally before using OSM geocoding."""

    known_places = {
        "majestic": (12.9767, 77.5713),
        "kempegowda bus station": (12.9767, 77.5713),
        "whitefield": (12.9698, 77.7500),
        "mg road": (12.9756, 77.6066),
        "indiranagar": (12.9784, 77.6408),
        "koramangala": (12.9352, 77.6245),
        "electronic city": (12.8399, 77.6770),
        "hebbal": (13.0358, 77.5970),
        "marathahalli": (12.9569, 77.7011),
        "jayanagar": (12.9250, 77.5838),
    }

    normalized = query.strip().casefold()

    # Avoid an external geocoding request for known locations.
    if normalized in known_places:
        return known_places[normalized]

    # Fall back to OpenStreetMap geocoding for other places.
    try:
        return ox.geocode(
            f"{query}, Bengaluru, Karnataka, India"
        )
    except Exception:
        return None



@st.cache_data(ttl=86400, show_spinner=False)
def resolve(query):
    """
    Resolve a location using:
    1. Binary Search on exact labels.
    2. Substring search on named road/junction labels.
    3. Geocoding followed by nearest road-node lookup.
    """

    query = query.strip()

    if not query:
        return None

    normalized_query = query.casefold()

    # Method 1: Exact Binary Search.
    index = binary_search(
        labels_lower,
        normalized_query,
    )

    if index >= 0:
        return label_to_node[labels[index]]

    # Method 2: Search for a matching named label.
    matches = [
        label
        for label in labels
        if normalized_query in label.casefold()
        and not label.startswith("Junction ")
    ]

    if matches:
        return label_to_node[matches[0]]

    # Method 3: Geocode the location.
    coordinates = geocode_location(query)

    if coordinates is None:
        return None

    latitude, longitude = coordinates

    # Find the nearest node in the actual loaded graph.
    try:
        node_id = ox.distance.nearest_nodes(
            G,
            X=longitude,
            Y=latitude,
        )

        return node_id

    except Exception as error:
        st.error(
            f"Could not locate '{query}' on the road graph: "
            f"{type(error).__name__}: {error}"
        )
        return None


# --------------------------------------------------
# SIDEBAR: ROUTE PLANNER
# --------------------------------------------------

with st.sidebar:
    st.header("Plan your journey")
    with st.form("route_planner_form"):
        st.caption("Enter two places in Bengaluru, then find a route.")

        start_q = st.text_input(
            "Starting point",
            "Majestic",
            help="Try Majestic, MG Road, Indiranagar, or another Bengaluru landmark.",
        )

        end_q = st.text_input(
            "Destination",
            "Whitefield",
            help="Enter a Bengaluru area, landmark, or road name.",
        )

        with st.expander("Advanced options", expanded=False):
            st.caption("Optional settings. You can find a route without filling in an incident report.")
            
            traffic_key = get_tomtom_key()


            use_live = st.checkbox(
                "Try to use TomTom live traffic",
                value=True,
                help="If the live feed fails, the app will use available road-speed estimates.",
            )

            incident_text = st.text_area(
                "Optional traffic incident report",
                placeholder="Example: Accident near MG Road; roadblock on a junction",
                help="Used to test incident keyword matching and incident-aware routing.",
            )

            closed_words = st.text_input(
                "Words that indicate a closure",
                "road closed, roadblock",
            )

        run = st.form_submit_button(
            "Find my route",
            type="primary",
            use_container_width=True,
        )

    st.caption("Traffic note: when live traffic is unavailable, the app uses estimated road speeds. It does not generate fake live traffic.")


    # --------------------------------------------------

# RESOLVE ORIGIN AND DESTINATION
# --------------------------------------------------

start = resolve(start_q)
end = resolve(end_q)


# --------------------------------------------------
# TRAFFIC DATA
# --------------------------------------------------

speeds = st.session_state.get("traffic_speeds", {})
traffic_status = st.session_state.get(
    "traffic_status", "Traffic not requested yet"
)
traffic_time = st.session_state.get("traffic_time")

if not use_live:
    speeds = {}
    traffic_status = "Live traffic disabled; using OSM posted speeds"
elif not traffic_key:
    speeds = {}
    traffic_status = "TomTom key not configured; using OSM posted speeds"


# --------------------------------------------------
# INCIDENT DETECTION
# --------------------------------------------------

closed = [
    word.strip().casefold()
    for word in closed_words.split(",")
    if word.strip()
]

incident_keywords = [
    "accident",
    "roadblock",
    "waterlogging",
    "vehicle breakdown",
    "road closed",
    "traffic jam",
]

incident_hits = (
    horspool_search(
        incident_text.casefold(),
        incident_keywords,
    )
    if incident_text.strip()
    else []
)

incident_active = bool(incident_hits)

@st.cache_resource(show_spinner="Applying incident closures...")
def cached_incident_graph(report, closure_csv):
    """Cache incident-adjusted graph copies for identical incident inputs."""
    closure_words = [
        word.strip().casefold()
        for word in closure_csv.split(",")
        if word.strip()
    ]
    return apply_incidents(G, report, closure_words)


if incident_active:
    Gwork = cached_incident_graph(incident_text, closed_words)
else:
    Gwork = G


# --------------------------------------------------
# TRAVEL-TIME COST FUNCTION
# --------------------------------------------------


def edge_cost(u, v, data):
    """Calculate estimated edge travel time in minutes."""

    length = float(data.get("length", 0.0))

    # If edge length is missing, calculate it from node coordinates.
    if length <= 0:
        from math import radians, sin, cos, asin, sqrt

        node_u = G.nodes[u]
        node_v = G.nodes[v]

        lat1 = radians(float(node_u["y"]))
        lon1 = radians(float(node_u["x"]))
        lat2 = radians(float(node_v["y"]))
        lon2 = radians(float(node_v["x"]))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            sin(dlat / 2) ** 2
            + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        )

        length = 2 * 6371000 * asin(sqrt(min(1.0, a)))

    speed = 30.0

    maxspeed = data.get("maxspeed_kph", data.get("speed_kph"))

    if maxspeed is not None:
        try:
            speed = float(str(maxspeed).split()[0])
        except (ValueError, TypeError):
            pass
    elif data.get("maxspeed") is not None:
        try:
            speed = float(str(data["maxspeed"]).split()[0])
        except (ValueError, TypeError):
            pass

    speed = max(5.0, min(speed, 100.0))

    # Convert distance and speed to minutes.
    minutes = (length / 1000.0) / speed * 60.0

    # Adjust for live traffic if available.
    ratio = speeds.get((str(u), str(v)))

    if ratio is not None:
        minutes /= max(0.12, min(1.0, float(ratio)))

    road_name = str(data.get("name", "")).casefold()

    if incident_active and any(
        word in road_name for word in closed
    ):
        return float("inf")

    return max(minutes, 0.001)



# --------------------------------------------------
# ROUTE CALCULATION
# --------------------------------------------------

if start is not None and end is not None:

    st.caption(
        f"Road network loaded: "
        f"{G.number_of_nodes():,} nodes · "
        f"{G.number_of_edges():,} directed edges"
    )

    saved_result = st.session_state.get("route_result")
    locations_changed = (
        saved_result is None
        or saved_result[3] != start
        or saved_result[4] != end
    )

    # Recalculate only on form submission or when resolved endpoints change.
    if run or locations_changed:
        # A newly calculated route invalidates alternatives from the prior result.
        st.session_state.pop("alternative_routes", None)
        start_time = time.perf_counter()

        # Clear stale measurements before calculating this request.
        speeds = {}
        if not use_live:
            traffic_status = (
                "Live traffic disabled; using OSM posted speeds"
            )
        elif not traffic_key:
            traffic_status = (
                "TomTom key not configured; using OSM posted speeds"
            )
        else:
            traffic_status = "Finding initial route before traffic probes"

        # First route uses the current baseline costs.
        route, cost = dijkstra(
            Gwork,
            start,
            end,
            edge_cost,
        )

        # Probe points along the initial route, then recalculate with
        # the live traffic ratios returned for those directed edges.
        if use_live and traffic_key and route:
            try:
                speeds, traffic_status, traffic_time = tomtom_edge_speeds(
                    G,
                    traffic_key,
                    route=route,
                )
            except Exception as error:
                speeds = {}
                traffic_status = (
                    f"TomTom route probes failed ({type(error).__name__}); "
                    "using OSM posted speeds"
                )
                traffic_time = None

            if speeds:
                route, cost = dijkstra(
                    Gwork,
                    start,
                    end,
                    edge_cost,
                )
        else:
            traffic_time = None

        elapsed = (time.perf_counter() - start_time) * 1000

        st.session_state["traffic_speeds"] = speeds
        st.session_state["traffic_status"] = traffic_status
        st.session_state["traffic_time"] = traffic_time
        st.session_state["route_result"] = (
            route,
            cost,
            elapsed,
            start,
            end,
        )

    route, cost, elapsed, saved_start, saved_end = (
        st.session_state["route_result"]
    )
    speeds = st.session_state.get("traffic_speeds", {})
    traffic_status = st.session_state.get(
        "traffic_status", traffic_status
    )
    traffic_time = st.session_state.get("traffic_time")

    col1, col2, col3 = st.columns(3)

    st.subheader("Your journey")
    st.caption(f"{start_q.strip()}  →  {end_q.strip()}")
    live_available = traffic_status.startswith("TomTom live") and bool(speeds)
    col1.metric("Traffic source", "Live data" if live_available else "Speed estimates")
    col2.metric("Incident matches", len(incident_hits))
    col3.metric("Road network", "OpenStreetMap")

    status_lower = traffic_status.casefold()
    if live_available:
        st.success("Live traffic data was received for some route segments. Coverage may be partial.")
    elif "401" in status_lower:
        st.warning("Live traffic is unavailable because TomTom rejected the API key (HTTP 401). Open Advanced options to check the key and API access. This route uses available road-speed estimates.")
    elif "403" in status_lower:
        st.warning("TomTom denied the traffic request (HTTP 403). Check that your key and subscription allow this API. This route uses available road-speed estimates.")
    elif "429" in status_lower:
        st.warning("TomTom request limit reached (HTTP 429). Wait before trying again. This route uses available road-speed estimates.")
    else:
        st.info("Live traffic is not available for this request. The route uses available road-speed estimates.")

    with st.expander("Technical traffic details", expanded=False):
        st.write("Status returned by traffic service:", traffic_status)
        st.write("Traffic speed entries received:", len(speeds))
        st.write("Sample traffic speed ratios:", list(speeds.items())[:5])
        route_edge_count = max(0, len(route) - 1)
        matched_route_edges = sum(
            (str(u), str(v)) in speeds
            for u, v in zip(route, route[1:])
        ) if route else 0
        st.write("Edges in selected route:", route_edge_count)
        st.write("Selected route edges with direct traffic ratios:", matched_route_edges)
        if traffic_time:
            st.write("Last successful traffic probe (UTC):", traffic_time)

    if not route or cost == float("inf"):

        st.error(
            "No feasible route found. Try different locations "
            "or review the incident report."
        )

    else:

        coords = route_geometry(G, route)

        # ------------------------------------------
        # ALTERNATIVE ROUTES
        # ------------------------------------------

        def alternative_edge_candidates(bundle):
            """Return individual edge dictionaries for Graph or MultiDiGraph."""
            if not isinstance(bundle, dict) or not bundle:
                return []
            if all(isinstance(value, dict) for value in bundle.values()):
                return list(bundle.values())
            return [bundle]

        def path_metrics(path):
            """Return estimated travel time (minutes) and road distance (km)."""
            total_minutes = 0.0
            total_metres = 0.0

            for u, v in zip(path, path[1:]):
                cost_edges = alternative_edge_candidates(
                    Gwork.get_edge_data(u, v) or {}
                )
                distance_edges = alternative_edge_candidates(
                    G.get_edge_data(u, v) or {}
                )

                edge_minutes = [
                    edge_cost(u, v, edge)
                    for edge in cost_edges
                ]
                edge_lengths = [
                    float(edge.get("length", 0.0))
                    for edge in distance_edges
                ]

                total_minutes += min(edge_minutes, default=float("inf"))
                total_metres += min(edge_lengths, default=0.0)

            return total_minutes, total_metres / 1000.0

        if st.button(
            "Find alternative routes",
            key="find_alternative_routes_v2",
        ):
            with st.spinner("Searching for alternative routes..."):
                alternatives = []
                penalized_edges = set(zip(route, route[1:]))

                # Re-run Dijkstra while penalizing edges already used.
                # This encourages distinct routes without changing the road graph.
                for _ in range(2):
                    def alternative_cost(u, v, data):
                        base_cost = edge_cost(u, v, data)
                        if (u, v) in penalized_edges:
                            return base_cost * 2.5
                        return base_cost

                    alt_path, alt_cost = dijkstra(
                        Gwork, start, end, alternative_cost
                    )

                    if (
                        not alt_path
                        or alt_cost == float("inf")
                        or alt_path == route
                        or alt_path in alternatives
                    ):
                        break

                    alternatives.append(alt_path)
                    penalized_edges.update(zip(alt_path, alt_path[1:]))

                st.session_state["alternative_routes"] = {
                    "endpoints": (start, end),
                    "routes": alternatives,
                }

        alternative_data = st.session_state.get("alternative_routes", {})
        alternative_paths = (
            alternative_data.get("routes", [])
            if alternative_data.get("endpoints") == (start, end)
            else []
        )

        if alternative_paths:
            st.subheader("Route comparison")

            comparison_rows = []
            recommended_minutes, recommended_km = path_metrics(route)
            comparison_rows.append({
                "Route": "Recommended route",
                "Estimated travel time (min)": round(recommended_minutes, 2),
                "Distance (km)": round(recommended_km, 2),
                "Road segments": len(route) - 1,
            })

            for index, alt_path in enumerate(alternative_paths, start=1):
                alt_minutes, alt_km = path_metrics(alt_path)
                comparison_rows.append({
                    "Route": f"Alternative {index}",
                    "Estimated travel time (min)": round(alt_minutes, 2),
                    "Distance (km)": round(alt_km, 2),
                    "Road segments": len(alt_path) - 1,
                })

            st.dataframe(
                pd.DataFrame(comparison_rows),
                use_container_width=True,
                hide_index=True,
            )
        elif alternative_data.get("endpoints") == (start, end):
            st.info(
                "No distinct alternative routes were found for this journey."
            )

        distance_metres = 0.0

        for u, v in zip(route, route[1:]):

            edge_data = G.get_edge_data(u, v) or {}

            # A MultiDiGraph may have multiple parallel edges.
            if edge_data and all(
                isinstance(value, dict)
                for value in edge_data.values()
            ):
                edge_data = min(
                    edge_data.values(),
                    key=lambda item: float(
                        item.get("length", 0)
                    ),
                )

            distance_metres += float(
                edge_data.get("length", 0)
            )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Estimated travel time",
            f"{cost:.1f} min",
        )

        col2.metric(
            "Road distance",
            f"{distance_metres / 1000:.1f} km",
        )

        col3.metric(
            "Route search time",
            f"{elapsed / 1000:.1f} sec" if elapsed >= 1000 else f"{elapsed:.0f} ms",
            help="Time spent calculating this route; graph loading and map display are separate.",
        )

        if coords:

            center = [
                sum(point[0] for point in coords) / len(coords),
                sum(point[1] for point in coords) / len(coords),
            ]

            map_view = folium.Map(
                location=center,
                zoom_start=12,
                tiles="OpenStreetMap",
            )

            # Draw alternatives first so the recommended traffic-coloured route
            # remains visible on top.
            alternative_data = st.session_state.get("alternative_routes", {})
            if alternative_data.get("endpoints") == (start, end):
                alternative_colors = ["#f59e0b", "#a78bfa"]

                for index, alt_path in enumerate(
                    alternative_data.get("routes", [])
                ):
                    alt_coords = route_geometry(G, alt_path)
                    if len(alt_coords) >= 2:
                        folium.PolyLine(
                            alt_coords,
                            color=alternative_colors[index % len(alternative_colors)],
                            weight=5,
                            opacity=0.8,
                            tooltip=f"Alternative route {index + 1}",
                        ).add_to(map_view)

            # Colour each recommended-route segment using available TomTom speed ratios.
            # Unknown segments remain cyan rather than being labelled as clear.
            for u, v, point_a, point_b in zip(
                route[:-1],
                route[1:],
                coords[:-1],
                coords[1:],
            ):
                ratio = speeds.get((str(u), str(v)))

                if ratio is None:
                    color = "#22d3ee"
                    label = "Live traffic data unavailable"
                elif ratio <= 0.40:
                    color = "#dc2626"
                    label = "Heavy congestion"
                elif ratio <= 0.70:
                    color = "#f59e0b"
                    label = "Moderate congestion"
                else:
                    color = "#16a34a"
                    label = "Relatively free-flowing"

                folium.PolyLine(
                    [point_a, point_b],
                    color=color,
                    weight=7,
                    opacity=0.95,
                    tooltip=f"{label} | Road segment {u} → {v}",
                ).add_to(map_view)

            folium.Marker(
                coords[0],
                tooltip="Origin",
                icon=folium.Icon(color="green"),
            ).add_to(map_view)

            folium.Marker(
                coords[-1],
                tooltip="Destination",
                icon=folium.Icon(color="red"),
            ).add_to(map_view)

            st_folium(
                map_view,
                use_container_width=True,
                height=530,
                key="optimized_route_map",
                returned_objects=[],
            )

        st.success("Route found on the OpenStreetMap road network.")
        st.caption("Travel time is an estimate, not a guaranteed arrival time. When live traffic is unavailable, it may not reflect current congestion.")

else:

    st.warning(
        "Could not resolve one or both locations. "
        "Check the spelling, internet connection, "
        "and OpenStreetMap geocoding availability."
    )




# --------------------------------------------------
# ROUTING ALGORITHM COMPARISON
# --------------------------------------------------

with st.expander("Compare routing algorithms", expanded=False):
    st.write(
        "Run three algorithms on the same road network and travel-time cost "
        "function. The comparison runs only when you press the button."
    )
    st.caption(
        "Execution times are measured locally and can vary with system load. "
        "All algorithms use the currently available traffic ratios and incident-adjusted network."
    )

    if start is None or end is None:
        st.info("Choose valid starting and destination locations first.")
    elif st.button("Run algorithm comparison", type="primary", key="run_algorithm_comparison"):
        def comparison_heuristic(node_a, node_b):
            """Straight-line lower bound in minutes, assuming at most 100 km/h."""
            from math import radians, sin, cos, asin, sqrt

            a = G.nodes[node_a]
            b = G.nodes[node_b]
            lat1, lon1 = radians(float(a["y"])), radians(float(a["x"]))
            lat2, lon2 = radians(float(b["y"])), radians(float(b["x"]))
            dlat, dlon = lat2 - lat1, lon2 - lon1
            h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            metres = 2 * 6371000 * asin(sqrt(min(1.0, h)))
            return (metres / 1000.0) / 100.0 * 60.0

        def route_distance_km(path):
            total = 0.0
            for u, v in zip(path, path[1:]):
                bundle = Gwork.get_edge_data(u, v) or {}
                candidates = list(bundle.values()) if (
                    bundle and all(isinstance(value, dict) for value in bundle.values())
                ) else [bundle]
                lengths = [
                    float(item.get("length", 0.0))
                    for item in candidates
                    if isinstance(item, dict)
                ]
                total += min(lengths) if lengths else 0.0
            return total / 1000.0

        algorithms = [
            ("Dijkstra", lambda: dijkstra_with_stats(Gwork, start, end, edge_cost)),
            ("A*", lambda: astar_search(
                Gwork, start, end, edge_cost, comparison_heuristic
            )),
            ("Bidirectional Dijkstra", lambda: bidirectional_dijkstra(
                Gwork, start, end, edge_cost
            )),
        ]

        results = []
        progress = st.progress(0, text="Preparing algorithm comparison...")
        for idx, (name, run_algorithm) in enumerate(algorithms, start=1):
            try:
                started_at = time.perf_counter()
                path, total_cost, expanded_nodes = run_algorithm()
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                if path and total_cost != float("inf"):
                    results.append({
                        "Algorithm": name,
                        "Travel time (min)": round(float(total_cost), 2),
                        "Distance (km)": round(route_distance_km(path), 2),
                        "Expanded nodes": int(expanded_nodes),
                        "Execution time (ms)": round(elapsed_ms, 2),
                        "Status": "Route found",
                    })
                else:
                    results.append({
                        "Algorithm": name,
                        "Travel time (min)": None,
                        "Distance (km)": None,
                        "Expanded nodes": int(expanded_nodes),
                        "Execution time (ms)": round(elapsed_ms, 2),
                        "Status": "No route",
                    })
            except Exception as error:
                results.append({
                    "Algorithm": name,
                    "Travel time (min)": None,
                    "Distance (km)": None,
                    "Expanded nodes": None,
                    "Execution time (ms)": None,
                    "Status": f"Error: {type(error).__name__}: {error}",
                })
            progress.progress(idx / len(algorithms), text=f"Completed {name}...")
        progress.empty()

        comparison_df = pd.DataFrame(results)
        st.session_state["algorithm_comparison_results"] = comparison_df

    comparison_df = st.session_state.get("algorithm_comparison_results")
    if comparison_df is not None:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # ------------------------------------------
        # VISUAL COMPARISON AND CORRECTNESS CHECK
        # ------------------------------------------

        valid_results = comparison_df[
            (comparison_df["Status"] == "Route found")
            & comparison_df["Execution time (ms)"].notna()
        ].copy()

        if not valid_results.empty:
            st.subheader("Performance comparison")

            # Find and highlight the fastest algorithm.
            fastest = valid_results.loc[
                valid_results["Execution time (ms)"].idxmin()
            ]

            st.metric(
                "Fastest algorithm in this run",
                fastest["Algorithm"],
                f'{fastest["Execution time (ms)"]:.2f} ms'
            )

            # Show execution-time and workload charts.
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.write("**Execution time (lower is better)**")
                time_chart = valid_results.set_index("Algorithm")[
                    ["Execution time (ms)"]
                ]
                st.bar_chart(time_chart)

            with chart_col2:
                st.write("**Expanded nodes (lower is better)**")
                nodes_chart = valid_results.set_index("Algorithm")[
                    ["Expanded nodes"]
                ]
                st.bar_chart(nodes_chart)

            st.write("**Route-cost correctness check**")

            reference_rows = valid_results[
                valid_results["Algorithm"] == "Dijkstra"
            ]

            if not reference_rows.empty:
                reference_cost = float(
                    reference_rows.iloc[0]["Travel time (min)"]
                )

                for _, result in valid_results.iterrows():
                    algorithm_name = result["Algorithm"]
                    algorithm_cost = float(result["Travel time (min)"])

                    # Values are displayed rounded to two decimal places.
                    difference = abs(algorithm_cost - reference_cost)

                    if difference <= 0.02:
                        st.success(
                            f"{algorithm_name}: route cost matches "
                            "Dijkstra within display-rounding tolerance."
                        )
                    else:
                        st.warning(
                            f"{algorithm_name}: route cost differs from "
                            f"Dijkstra by {difference:.2f} minutes. "
                            "Investigate before claiming equivalent optimality."
                        )
            else:
                st.info(
                    "Dijkstra did not return a route, so a comparison "
                    "against its reference cost is unavailable."
                )

            st.caption(
                "Results apply to this particular route and run. "
                "Execution times vary with system load. Matching rounded "
                "costs are evidence of agreement, not a formal proof of optimality."
            )
        else:
            st.info("No successful algorithm results are available to chart.")

        st.caption(
            "Expanded nodes counts the nodes removed from each algorithm's priority queue "
            "and processed. This is a practical workload measure, not a hardware-independent benchmark."
        )


# --------------------------------------------------
# ALGORITHM LABORATORY
# --------------------------------------------------

with st.expander("Algorithm laboratory"):

    st.write(
        "**Binary Search:** exact lookup over sorted "
        "road/junction labels."
    )

    search = st.text_input(
        "Search a location",
        key="algo_search",
    )

    if search:

        index = binary_search(
            labels_lower,
            search.casefold(),
        )

        if index >= 0:
            st.write(f"Exact match: {labels[index]}")
        else:
            st.write(
                "No exact match. Try a full road/junction label "
                "or use the route planner's geographic lookup."
            )

    st.write(
        "**Horspool:** keyword search for incident text."
    )

    st.write(
        "Detected keywords: "
        + (
            ", ".join(incident_hits)
            if incident_hits
            else "None detected"
        )
    )

    st.write(
        "**Dijkstra:** minimum estimated travel-time path "
        "with non-negative road costs."
    )


# --------------------------------------------------
# RSA INCIDENT SECURITY
# --------------------------------------------------

with st.expander("RSA incident security"):

    message = st.text_area(
        "Sensitive incident message",
        key="rsa_message",
        value="Traffic incident: lane obstruction near junction.",
    )

    if st.button("Generate key and encrypt message"):

        box = RSABox()
        encrypted = box.encrypt(message)

        st.session_state["rsa_demo"] = (
            box,
            encrypted,
        )

    if "rsa_demo" in st.session_state:

        box, encrypted = st.session_state["rsa_demo"]

        st.write("Encrypted message:")

        st.code(
            encrypted,
            language="text",
        )

        if st.button("Decrypt with private key"):

            try:
                decrypted = box.decrypt(encrypted)
                st.success(decrypted)

            except Exception as error:
                st.error(
                    f"Decryption failed: {error}"
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    "Data integrity note: RSA demonstration keys are generated "
    "in memory for this session. This is a project demo, not a "
    "production key-management system."
)
