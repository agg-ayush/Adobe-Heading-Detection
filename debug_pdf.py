#!/usr/bin/env python3
import fitz

doc = fitz.open('dataset/pdfs/file02.pdf')
for page_num in range(min(3, len(doc))):
    page = doc[page_num]
    text_dict = page.get_text('dict')
    print(f'=== PAGE {page_num + 1} ===')
    for block in text_dict['blocks']:
        if block['type'] == 0:
            for line in block['lines']:
                for span in line['spans']:
                    if span['size'] > 11:
                        print(f"Size: {span['size']:.1f} | Text: {span['text'][:80]}")
    print()
