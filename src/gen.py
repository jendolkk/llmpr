import os
import sys

from ollama import chat
import json
# response = chat(model='captioner', messages=[
#   {
#     'role': 'user',
#     'content': 'analyse this picture',
#     'images': ['D:\\wyh\\data\\2016\\Segmentednew\\USD0746544-20160105-D00002_2.png']
#   },
# ])
# print(response.message.content)
beg = 0
for subdir in os.walk('D:\\wyh\\data\\2016\\desc'):
  print(subdir)
  if len(subdir[2]) > 0:
    beg = (1 + int(subdir[2][-1][:-5])) * 1000
captions = []
for subdir in os.walk('D:\\wyh\\data\\2016\\Segmentednew'):
  print(len(subdir[2]))
  # print(subdir[0], subdir[1], subdir[2][0])
  path = subdir[0]
  for i, img in enumerate(subdir[2]):
    if i < beg:
      continue
    if int(img[27:-4]) > 100:
      continue
    print(i)
    response = chat(model='captioner', messages=[
      {
        'role': 'user',
        'content': 'Analyze the provided patent image and generate a concise description focusing on the unique features that distinguish it from other patents. Highlight key structural, functional, or design elements without speculative interpretation. Limit the description to 500 words, ensuring clarity and precision in identifying the patent\'s distinctive characteristics.',
        'images': [os.path.join(path, img)]
      },
    ])
    # name, extension = os.path.splitext(img)
    caption = {
      "patentID": img[:19],
      "figid": img[27:-4],
      "caption": response.message.content,
      "subfigure_file": img,
    }
    captions.append(caption)

    if (i + 1) % 1000 == 0:
      print(i + 1)
      with open(os.path.join('D:\\wyh\\data\\2016\\desc', str(i // 1000) + '.json'), mode='w', encoding='utf-8') as f:
        f.write(json.dumps(captions, indent=2))
      captions = []
      if i == 9999:
        sys.exit(0)