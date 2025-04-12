import sys
from bisect import bisect_right

import constant
import json
from collections import defaultdict

file = constant.json_file
print(file)

with open(file, 'r') as f:
  global a
  a = json.load(f)
  n = 0
  summ = 0
  mp = {}
  for d in a:
    id = int(d["classification_locarno"])
    n = max(n, id)
    title = d["object_title"].strip().lower()
    if title not in mp:
      d["title_id"] = len(mp)
      mp[title] = len(mp)
    else:
      d["title_id"] = mp[title]
  print(f"max locarno id: {n}")

  used = set()
  b = [0] * (n + 1)
  for d in a:
    id = int(d["classification_locarno"])
    if id not in used:
      b[id] += 1
      summ += 1
  c = [0] * (n + 1)
  for i in range(n + 1):
    c[i] = i
  c.sort(key=lambda x: -b[x])
  id = 0
  s = 0
  for i in range(n + 1):
    if s > summ * 0.4:
      id = i
      break
    s += b[c[i]]
  for d in a:
    id = int(d["classification_locarno"])
    if c[id] < id:
      d["head"] = 1
    else:
      d["head"] = 0
  print(f"sum(b): {sum(b)}")

# sys.exit(0)
with open(constant.ok_file, 'w') as f:
  json.dump(a, f, indent=2)
