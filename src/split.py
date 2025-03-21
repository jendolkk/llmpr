import json
import constant

origin = constant.ok_file
train = constant.train_json
val = constant.val_json
test = constant.test_json

i = 0
j = 0
with open(origin, 'r', encoding='utf-8') as f:
  a = json.load(f)
  sz = len(a)

  i = int(sz * 0.8)
  j = i + int(sz * 0.1)
with open(train, 'w', encoding='utf-8') as f:
  json.dump(a[:i], f, indent=2)
with open(val, 'w', encoding='utf-8') as f:
  json.dump(a[i:j], f, indent=2)
with open(test, 'w', encoding='utf-8') as f:
  json.dump(a[j:], f, indent=2)
