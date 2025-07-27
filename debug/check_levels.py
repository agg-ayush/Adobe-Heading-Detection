import json

for i in range(1, 7):
    with open(f'dataset/outputs/file0{i}.json', 'r') as f:
        data = json.load(f)
    h1_count = len([x for x in data['outline'] if x['level'] == 'H1'])
    h2_count = len([x for x in data['outline'] if x['level'] == 'H2'])
    h3_count = len([x for x in data['outline'] if x['level'] == 'H3'])
    h4_count = len([x for x in data['outline'] if x.get('level') == 'H4'])
    print(f'File {i}: H1={h1_count}, H2={h2_count}, H3={h3_count}, H4={h4_count}, Total={len(data["outline"])}, Title="{data["title"]}"')
