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
  used = set()
  for d in a:
    id = int(d["classification_locarno"])
    if id not in used:
      n = max(n, id)
      used.add(id)
  b = [0] * (n + 1)
  # used = set()
  used.clear()
  print(n)
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
    if s >= summ * 0.4:
      id = i
      break
    s += b[c[i]]
  for d in a:
    id = int(d["classification_locarno"])
    # print(id)
    if c[id] <= id:
      d["head"] = 1

  print(sum(b))
# sys.exit(0)
with open(constant.ok_file, 'w') as f:
  json.dump(a, f, indent=2)
