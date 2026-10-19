"""Optional TomTom live traffic adapter. No synthetic traffic fallback is created."""
import requests
from datetime import datetime, timezone

def tomtom_edge_speeds(graph, api_key, timeout=8):
    """Probe representative graph nodes. Returns edge congestion ratios and explicit source status.
    TomTom flow probes are point-based; mapping to road edges is approximate, not segment-perfect.
    """
    nodes=list(graph.nodes(data=True))
    if not nodes: return {}, "TomTom live traffic unavailable: empty graph", None
    # Sample spatially across the graph, bounded to avoid excessive API calls.
    stride=max(1,len(nodes)//20)
    sample=nodes[::stride][:20]
    ratios={}
    successes=0
    stamp=None
    for node,data in sample:
        if "y" not in data or "x" not in data: continue
        url=f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        try:
            r=requests.get(url,params={"point":f"{data['y']},{data['x']}","key":api_key},timeout=timeout)
            r.raise_for_status(); payload=r.json()["flowSegmentData"]
            current=float(payload["currentSpeed"]); free=float(payload["freeFlowSpeed"])
            ratio=max(.12,min(1.0,current/free)) if free>0 else 1.0
            successors=list(graph.successors(node))
            for v in successors:
                ratios[(str(node),str(v))]=ratio
            successes+=1
            stamp=datetime.now(timezone.utc).isoformat()
        except Exception:
            continue
    if not successes:
        return {}, "TomTom live traffic unavailable; using OpenStreetMap posted speeds", None
    return ratios, f"TomTom live traffic: {successes} successful point probes (edge mapping approximate)", stamp

def apply_incidents(graph, report, closure_keywords):
    """Return a copy; only closes roads when a closure keyword and road name can be linked.
    Does not invent incident locations or modify the original network.
    """
    g=graph.copy()
    lower=report.casefold()
    closures=[w for w in closure_keywords if w and w in lower]
    if not closures: return g
    # Only close named roads explicitly mentioned in the report; no global closure guesses.
    for u,v,key,data in list(g.edges(keys=True,data=True)):
        name=str(data.get("name","")).casefold()
        if name and name in lower and any(w in lower for w in closures):
            g.remove_edge(u,v,key)
    return g

def estimate_edge_minutes(length_m, speed_kmh):
    return (float(length_m)/1000.0)/max(float(speed_kmh),1.0)*60.0
