# Bengaluru SmartRoute AI
An AI-assisted traffic-routing demonstration using a real OpenStreetMap road graph, Dijkstra shortest path, Binary Search, Horspool string matching, optional TomTom live traffic, and RSA-OAEP incident-message encryption.

## Honest data policy
- Road geometry comes from OpenStreetMap when you run `build_graph.py`.
- TomTom traffic is optional and requires your own API key.
- If the feed is missing or unavailable, the app explicitly reports that it is using posted OSM speeds. It does not generate fake live traffic.
- Traffic flow probes are sampled and mapped approximately to nearby outgoing edges. They are not a substitute for a complete per-road live traffic feed.
- Incident text matching detects keywords; it does not verify incidents with authorities or infer a precise location from a report.

## Run locally
Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python build_graph.py
streamlit run app.py
```

The first graph download may take several minutes and needs internet access. The full Bengaluru drive network can be large.

## TomTom traffic
Set `TOMTOM_API_KEY` as an environment variable or add it to Streamlit Cloud Secrets. Never commit a real key to GitHub. A free API tier, quotas, and eligible endpoints can change; check your TomTom developer account before relying on availability.

## Deploy on Streamlit Community Cloud
1. Push this repository to GitHub.
2. Ensure `app.py`, `algos.py`, `traffic.py`, `build_graph.py`, and `requirements.txt` are in the repository root.
3. Create the Streamlit app with `app.py` as the entry point.
4. Add `TOMTOM_API_KEY` under app Secrets if you have a key.
5. Build the road graph before deployment and include `data/graph.graphml` only if repository size limits allow it. Otherwise use a persistent/external artifact location or a deployment build step. Do not expect Streamlit Cloud to preserve files generated interactively at runtime.
6. Verify that the status panel reports the actual data source. Never present posted-speed routing as live traffic.

## Algorithms
- Binary Search: exact search through sorted node labels, O(log n).
- Dijkstra: shortest estimated travel-time path with nonnegative edge costs, O((V+E) log V) with a binary heap.
- Horspool: searches incident reports for relevant keywords.
- RSA-OAEP: encrypts short incident messages. The demonstration generates an in-memory key pair; keys are not persisted.
- AI extension: train a speed/delay estimator only when suitable real observations and labels are available. This baseline does not claim a trained ML model.

## Tests
```bash
pytest -q
```

## Current limitations
- OSM node names are often missing; location search may require a distinctive named junction/road.
- OSM posted speeds are incomplete in some areas.
- The TomTom adapter samples up to 20 locations and uses approximate edge mapping.
- Incident closures are only applied when a road name is explicitly present in the report and can be matched.
- RSA-OAEP has message-size limits; use hybrid encryption (AES-GCM + RSA-wrapped key) for longer production messages.
