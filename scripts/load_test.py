"""Send random prediction requests to the API so Prometheus/Grafana have traffic to show.

Usage:  python scripts/load_test.py --url http://localhost:8000 --n 300
"""
import argparse
import json
import random
import time
import urllib.request

# Rough measurement ranges per species, so the traffic covers all three classes.
RANGES = {
    "setosa": [(4.6, 5.6), (3.0, 4.0), (1.2, 1.7), (0.1, 0.4)],
    "versicolor": [(5.5, 6.7), (2.3, 3.0), (3.8, 4.8), (1.1, 1.5)],
    "virginica": [(6.2, 7.6), (2.7, 3.3), (5.1, 6.4), (1.8, 2.4)],
}
KEYS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--delay", type=float, default=0.05)
    args = p.parse_args()

    counts = {}
    for i in range(args.n):
        species = random.choice(list(RANGES))
        body = {k: round(random.uniform(*r), 1) for k, r in zip(KEYS, RANGES[species])}
        req = urllib.request.Request(f"{args.url}/predict", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        result = json.loads(urllib.request.urlopen(req).read())
        counts[result["species"]] = counts.get(result["species"], 0) + 1
        if i % 10 == 0:
            urllib.request.urlopen(f"{args.url}/health").read()
        time.sleep(args.delay)
    print(f"Sent {args.n} prediction requests. Predicted: {counts}")


if __name__ == "__main__":
    main()
