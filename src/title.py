import json
import constant

with open(constant.ok_file, 'r') as f:
    global a
    a = json.load(f)
    mp = {}
    for d in a:
        name = d['object_title']
        if name not in mp:
            mp[name] = len(mp) + 1
    for d in a:
        name = d['object_title']
        d['title_id'] = mp[name]

with open(constant.ok_file, 'w') as f:
    json.dump(a, f)