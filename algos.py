"""Core algorithms for Bengaluru SmartRoute AI."""
import base64, heapq, json
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

def binary_search(sorted_values, target):
    """Return index of exact target in ascending sorted_values, else -1."""
    lo, hi = 0, len(sorted_values)-1
    while lo <= hi:
        mid=(lo+hi)//2
        if sorted_values[mid] == target: return mid
        if sorted_values[mid] < target: lo=mid+1
        else: hi=mid-1
    return -1

def horspool_search(text, patterns):
    """Return incident keywords found in text using Horspool pattern matching."""
    found=[]
    for pattern in patterns:
        p=pattern.casefold(); t=text.casefold()
        if not p: continue
        shift={p[i]:len(p)-1-i for i in range(len(p)-1)}
        i=len(p)-1; matched=False
        while i < len(t):
            k=0
            while k < len(p) and p[len(p)-1-k] == t[i-k]: k+=1
            if k==len(p): matched=True; break
            i += shift.get(t[i],len(p))
        if matched: found.append(pattern)
    return found

def dijkstra(graph, source, target, cost_fn):
    """Dijkstra for MultiDiGraph/DiGraph; cost_fn(u,v,edge_data) returns nonnegative cost."""
    dist={source:0.0}; prev={}; heap=[(0.0,source)]
    while heap:
        cost,u=heapq.heappop(heap)
        if cost != dist.get(u): continue
        if u==target: break
        for v,edge_bundle in graph[u].items():
            candidates = _edge_candidates(edge_bundle)
            weight = min((cost_fn(u, v, d) for d in candidates), default=float("inf"))
            if weight < 0: raise ValueError("Dijkstra requires non-negative edge weights")
            nd=cost+weight
            if nd < dist.get(v,float("inf")):
                dist[v]=nd; prev[v]=u; heapq.heappush(heap,(nd,v))
    if target not in dist: return [],float("inf")
    path=[target]
    while path[-1]!=source: path.append(prev[path[-1]])
    path.reverse()
    return path,dist[target]

def route_geometry(graph, route):
    return [(float(graph.nodes[n]["y"]),float(graph.nodes[n]["x"])) for n in route
            if "y" in graph.nodes[n] and "x" in graph.nodes[n]]

class RSABox:
    """RSA-OAEP demo for short messages; production systems should use hybrid encryption."""
    def __init__(self):
        self.private_key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        self.public_key=self.private_key.public_key()
    def encrypt(self,message):
        raw=self.public_key.encrypt(message.encode("utf-8"),padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(),label=None))
        return base64.b64encode(raw).decode("ascii")
    def decrypt(self,ciphertext):
        raw=base64.b64decode(ciphertext)
        return self.private_key.decrypt(raw,padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(),label=None)).decode("utf-8")
    def public_pem(self):
        return self.public_key.public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo).decode()


def _edge_candidates(edge_bundle):
    """Yield edge dictionaries for simple and multi-edge NetworkX graphs."""
    if (hasattr(edge_bundle, "values") and len(edge_bundle) > 0
            and all(isinstance(value, dict) for value in edge_bundle.values())):
        return edge_bundle.values()
    return [edge_bundle]


def dijkstra_with_stats(graph, source, target, cost_fn):
    """Dijkstra returning (path, total_cost, expanded_nodes)."""
    distances = {source: 0.0}
    previous = {}
    heap = [(0.0, source)]
    expanded = 0
    while heap:
        cost, u = heapq.heappop(heap)
        if cost != distances.get(u):
            continue
        expanded += 1
        if u == target:
            break
        for v, bundle in graph[u].items():
            weight = min((cost_fn(u, v, data) for data in _edge_candidates(bundle)), default=float("inf"))
            if weight < 0:
                raise ValueError("Dijkstra requires non-negative edge weights")
            candidate = cost + weight
            if candidate < distances.get(v, float("inf")):
                distances[v] = candidate
                previous[v] = u
                heapq.heappush(heap, (candidate, v))
    if target not in distances:
        return [], float("inf"), expanded
    path = [target]
    while path[-1] != source:
        path.append(previous[path[-1]])
    path.reverse()
    return path, distances[target], expanded


def astar_search(graph, source, target, cost_fn, heuristic_fn):
    """A* returning (path, total_cost, expanded_nodes)."""
    g_score = {source: 0.0}
    previous = {}
    heap = [(float(heuristic_fn(source, target)), 0.0, source)]
    expanded = 0
    closed = set()
    while heap:
        _, cost, u = heapq.heappop(heap)
        if cost != g_score.get(u) or u in closed:
            continue
        expanded += 1
        if u == target:
            break
        closed.add(u)
        for v, bundle in graph[u].items():
            weight = min((cost_fn(u, v, data) for data in _edge_candidates(bundle)), default=float("inf"))
            if weight < 0:
                raise ValueError("A* requires non-negative edge weights")
            candidate = cost + weight
            if candidate < g_score.get(v, float("inf")):
                g_score[v] = candidate
                previous[v] = u
                heapq.heappush(heap, (candidate + float(heuristic_fn(v, target)), candidate, v))
    if target not in g_score:
        return [], float("inf"), expanded
    path = [target]
    while path[-1] != source:
        path.append(previous[path[-1]])
    path.reverse()
    return path, g_score[target], expanded


def bidirectional_dijkstra(graph, source, target, cost_fn):
    """Bidirectional Dijkstra for directed graphs; returns (path, cost, expanded_nodes)."""
    if source == target:
        return [source], 0.0, 1
    forward_dist = {source: 0.0}
    backward_dist = {target: 0.0}
    forward_prev = {}
    backward_next = {}
    forward_heap = [(0.0, source)]
    backward_heap = [(0.0, target)]
    best = float("inf")
    meeting = None
    expanded = 0

    def clean(heap, distances):
        while heap and heap[0][0] != distances.get(heap[0][1]):
            heapq.heappop(heap)

    while forward_heap and backward_heap:
        clean(forward_heap, forward_dist)
        clean(backward_heap, backward_dist)
        if not forward_heap or not backward_heap:
            break
        if forward_heap[0][0] + backward_heap[0][0] >= best:
            break

        if forward_heap[0][0] <= backward_heap[0][0]:
            cost, u = heapq.heappop(forward_heap)
            if cost != forward_dist.get(u):
                continue
            expanded += 1
            if u in backward_dist and cost + backward_dist[u] < best:
                best, meeting = cost + backward_dist[u], u
            for v, bundle in graph[u].items():
                weight = min((cost_fn(u, v, data) for data in _edge_candidates(bundle)), default=float("inf"))
                if weight < 0:
                    raise ValueError("Dijkstra requires non-negative edge weights")
                candidate = cost + weight
                if candidate < forward_dist.get(v, float("inf")):
                    forward_dist[v] = candidate
                    forward_prev[v] = u
                    heapq.heappush(forward_heap, (candidate, v))
                if v in backward_dist and candidate + backward_dist[v] < best:
                    best, meeting = candidate + backward_dist[v], v
        else:
            cost, u = heapq.heappop(backward_heap)
            if cost != backward_dist.get(u):
                continue
            expanded += 1
            if u in forward_dist and cost + forward_dist[u] < best:
                best, meeting = cost + forward_dist[u], u
            # Reverse search follows original incoming edges p -> u.
            predecessors = graph.pred[u] if hasattr(graph, "pred") else graph
            for p, bundle in predecessors.items():
                if hasattr(graph, "pred"):
                    weight = min((cost_fn(p, u, data) for data in _edge_candidates(bundle)), default=float("inf"))
                else:
                    weight = min((cost_fn(p, u, data) for data in _edge_candidates(graph.get_edge_data(p, u, {}))), default=float("inf"))
                if weight < 0:
                    raise ValueError("Dijkstra requires non-negative edge weights")
                candidate = cost + weight
                if candidate < backward_dist.get(p, float("inf")):
                    backward_dist[p] = candidate
                    backward_next[p] = u
                    heapq.heappush(backward_heap, (candidate, p))
                if p in forward_dist and candidate + forward_dist[p] < best:
                    best, meeting = candidate + forward_dist[p], p

    if meeting is None or best == float("inf"):
        return [], float("inf"), expanded
    left = [meeting]
    while left[-1] != source:
        left.append(forward_prev[left[-1]])
    left.reverse()
    right = []
    cursor = meeting
    while cursor != target:
        cursor = backward_next[cursor]
        right.append(cursor)
    return left + right, best, expanded
