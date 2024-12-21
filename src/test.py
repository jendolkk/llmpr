import json
import constant
from transformers import AutoTokenizer, CLIPModel

data = json.load(constant.json_file) # list[dict]

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
# text_features = model.get_text_features(**inputs)

new_data = []
for mp in data:
  inputs = tokenizer([mp['caption']], padding=True, return_tensors='pt')
  mp['embedding'] = model.get_text_features(**inputs).squeeze(0)
  new_data.append(mp)

with open('./final.json') as f:
  f.write(json.dumps(new_data, indent=2))
