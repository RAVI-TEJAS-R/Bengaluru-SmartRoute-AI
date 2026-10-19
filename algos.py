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
            candidates=edge_bundle.values() if isinstance(edge_bundle,dict) and edge_bundle and all(isinstance(x,dict) for x in edge_bundle.values()) else [edge_bundle]
            weight=min((cost_fn(u,v,d) for d in candidates), default=float("inf"))
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
