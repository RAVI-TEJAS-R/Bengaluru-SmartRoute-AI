import os, time
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import st_folium
from algos import binary_search, dijkstra, horspool_search, RSABox, route_geometry
from traffic import tomtom_edge_speeds, apply_incidents, estimate_edge_minutes

st.set_page_config(page_title="Bengaluru SmartRoute AI", page_icon="🚦", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.3rem;max-width:1500px}
.hero{padding:1.4rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#101827,#163b50);color:white;margin-bottom:1rem}
.hero h1{margin:0;color:white}.hero p{color:#cbd5e1;margin:.4rem 0 0}
[data-testid="stMetric"]{background:rgba(120,140,160,.08);padding:12px;border-radius:12px}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🚦 Bengaluru SmartRoute AI</h1><p>Road-network routing · traffic-aware costs · incident matching · RSA protection</p></div>', unsafe_allow_html=True)

GRAPH_PATH="data/graph.graphml"
@st.cache_resource
def load_graph():
    if not os.path.exists(GRAPH_PATH):
        return None
    try:
        g=nx.read_graphml(GRAPH_PATH, node_type=int)
        return nx.MultiDiGraph(g)
    except Exception:
        try: return nx.read_graphml(GRAPH_PATH)
        except Exception: return None

G=load_graph()
if G is None:
    st.warning("Road network not built yet. Create it using the instructions below, then redeploy/reload.")
    st.code("python build_graph.py", language="bash")
    st.markdown("[Open project README](https://github.com/) for setup instructions.")
    st.stop()

# Normalize OSM graph values; location search uses sorted, human-readable labels.
nodes=list(G.nodes(data=True))
def node_label(item):
    n,d=item
    name=str(d.get("name","")).split(",")[0].strip()
    return f"{name} · {n}" if name else f"Junction {n}"
labels=sorted([node_label(x) for x in nodes], key=str.casefold)
label_to_node={node_label((n,d)):n for n,d in nodes}
labels=sorted(label_to_node, key=str.casefold)
labels_lower=[x.casefold() for x in labels]

def nearest_node(query):
    # Exact binary search first; geospatial nearest-node selection is handled by coordinates if available.
    i=binary_search(labels_lower, query.casefold())
    if i>=0: return label_to_node[labels[i]]
    # Name search fallback is explicitly a prefix/substring suggestion, not binary search.
    matches=[x for x in labels if query.casefold() in x.casefold()]
    return label_to_node[matches[0]] if matches else None

with st.sidebar:
    st.header("Route planner")
    start_q=st.text_input("Origin", "Majestic")
    end_q=st.text_input("Destination", "Whitefield")
    traffic_key=st.text_input("TomTom API key", value=os.getenv("TOMTOM_API_KEY",""), type="password", help="Optional. Add TOMTOM_API_KEY in Streamlit Secrets for deployment.")
    use_live=st.checkbox("Use TomTom live traffic when available", value=True)
    incident_text=st.text_area("Traffic incident report", placeholder="Example: Accident reported near junction; roadblock on affected road")
    closed_words=st.text_input("Closure keywords", "road closed, roadblock")
    run=st.button("Find optimized route", type="primary", use_container_width=True)
    st.caption("No synthetic live traffic is generated. If the live feed is unavailable, routing uses posted road speeds and labels that limitation.")

def resolve(q):
    q=q.strip().casefold()
    if not q: return None
    # Search actual OSM node names first; use binary search for exact sorted labels.
    exact=binary_search(labels_lower,q)
    if exact>=0: return label_to_node[labels[exact]]
    hits=[s for s in labels if q in s.casefold()]
    return label_to_node[hits[0]] if hits else None

start=resolve(start_q); end=resolve(end_q)
speeds, traffic_status, traffic_time = ({}, "Not requested", None)
if use_live and traffic_key:
    try:
        speeds, traffic_status, traffic_time=tomtom_edge_speeds(G, traffic_key)
    except Exception as e:
        traffic_status=f"Live feed unavailable ({type(e).__name__}); using OSM posted speeds"
elif use_live:
    traffic_status="TomTom key not configured; using OSM posted speeds"
else:
    traffic_status="Live traffic disabled; using OSM posted speeds"

closed=[x.strip().casefold() for x in closed_words.split(",") if x.strip()]
incident_hits=horspool_search(incident_text.casefold(), ["accident","roadblock","waterlogging","vehicle breakdown","road closed","traffic jam"]) if incident_text.strip() else []
incident_active=bool(incident_hits)
if incident_active:
    Gwork=apply_incidents(G, incident_text, closed)
else: Gwork=G

def edge_cost(u,v,data):
    length=float(data.get("length",1.0))
    speed=float(data.get("maxspeed_kph",data.get("speed_kph",30)))
    if isinstance(data.get("maxspeed"),str):
        try: speed=float(str(data["maxspeed"]).split()[0])
        except Exception: pass
    speed=max(5.0,min(speed,100.0))
    base=length/1000.0/speed*60
    ratio=speeds.get((str(u),str(v)))
    if ratio is not None: base/=max(.12,min(1.0,float(ratio)))
    name=str(data.get("name","")).casefold()
    if incident_active and any(w in name for w in closed): return float("inf")
    return max(base,0.001)

if start is not None and end is not None:
    st.caption(f"Road network loaded: {G.number_of_nodes():,} nodes · {G.number_of_edges():,} directed edges")
    c1,c2,c3=st.columns(3)
    c1.metric("Traffic data", "Live feed" if traffic_status.startswith("TomTom live") else "Fallback / unavailable")
    c2.metric("Incident keywords", len(incident_hits))
    c3.metric("Network", "OpenStreetMap")
    st.info(f"Traffic status: {traffic_status}")
    if run or "route_result" not in st.session_state:
        t0=time.perf_counter()
        route,cost=dijkstra(Gwork,start,end,edge_cost)
        elapsed=(time.perf_counter()-t0)*1000
        st.session_state["route_result"]=(route,cost,elapsed)
    route,cost,elapsed=st.session_state["route_result"]
    if not route or cost==float("inf"):
        st.error("No feasible route found. Try different locations or review the closure report.")
    else:
        coords=route_geometry(G,route)
        dist=0
        for u,v in zip(route,route[1:]):
            dat=G.get_edge_data(u,v) or {}
            dat=next(iter(dat.values())) if isinstance(dat,dict) and dat and isinstance(next(iter(dat.values())),dict) else dat
            dist+=float(dat.get("length",0))
        a,b,c=st.columns(3)
        a.metric("Estimated travel time", f"{cost:.1f} min")
        b.metric("Road distance", f"{dist/1000:.1f} km")
        c.metric("Route calculation", f"{elapsed:.2f} ms")
        if coords:
            center=[sum(p[0] for p in coords)/len(coords),sum(p[1] for p in coords)/len(coords)]
            m=folium.Map(location=center,zoom_start=12,tiles="CartoDB dark_matter")
            folium.PolyLine(coords,color="#22d3ee",weight=6,opacity=.9,tooltip="Optimized route").add_to(m)
            folium.Marker(coords[0],tooltip="Origin",icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(coords[-1],tooltip="Destination",icon=folium.Icon(color="red")).add_to(m)
            st_folium(m,use_container_width=True,height=530)
        st.success("Route calculated from the loaded OpenStreetMap road graph.")
else:
    st.warning("Could not match one or both locations to road-network node names. Try a junction/road name that exists in the loaded map.")

with st.expander("Algorithm laboratory"):
    st.write("**Binary Search:** exact lookup over sorted road/junction labels.")
    search=st.text_input("Search a location", key="algo_search")
    if search:
        i=binary_search(labels_lower,search.casefold())
        st.write(f"Exact match: {labels[i]}" if i>=0 else "No exact match. Use a distinctive part of a road/junction name.")
    st.write("**Horspool:** keyword search for incident text.")
    st.write("Detected keywords: "+(", ".join(incident_hits) if incident_hits else "None detected"))
    st.write("**Dijkstra:** minimum estimated travel-time path with non-negative road costs.")

with st.expander("RSA incident security"):
    msg=st.text_area("Sensitive incident message", key="rsa_message", value="Traffic incident: lane obstruction near junction.")
    if st.button("Generate key and encrypt message"):
        box=RSABox()
        encrypted=box.encrypt(msg)
        st.session_state["rsa_demo"]=(box,encrypted)
    if "rsa_demo" in st.session_state:
        box,encrypted=st.session_state["rsa_demo"]
        st.code(encrypted, language="text")
        if st.button("Decrypt with private key"):
            st.success(box.decrypt(encrypted))

st.caption("Data integrity note: RSA demonstration keys are generated in memory for this session. This is a project demo, not a production key-management system.")
