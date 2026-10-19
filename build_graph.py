"""Download a real OpenStreetMap drive network for Bengaluru and save GraphML."""
import os
import osmnx as ox

PLACE="Bengaluru, Karnataka, India"
OUT="data/graph.graphml"
def main():
    os.makedirs("data",exist_ok=True)
    ox.settings.use_cache=True
    ox.settings.log_console=True
    print(f"Downloading real OSM drive network for {PLACE} ...")
    graph=ox.graph_from_place(PLACE,network_type="drive",simplify=True,retain_all=False)
    graph=ox.add_edge_speeds(graph)
    graph=ox.add_edge_travel_times(graph)
    ox.save_graphml(graph,OUT)
    print(f"Saved {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges to {OUT}")
if __name__=="__main__": main()
