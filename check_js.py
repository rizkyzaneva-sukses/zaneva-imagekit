import re

with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

start = content.find('<script>')
end = content.find('</script>')
script = content[start+8:end]

# Find all getElementById calls
ids_in_js = re.findall(r"getElementById\(['\"]([^'\"]+)['\"]\)", script)
ids_in_js = list(set(ids_in_js))

# Find all id= in HTML (before script)
html = content[:start]
ids_in_html = set(re.findall(r'id=["\']([^"\']+)["\']', html))

missing = [x for x in ids_in_js if x not in ids_in_html]
print(f'IDs referenced in JS: {len(ids_in_js)}')
print(f'IDs in HTML: {len(ids_in_html)}')
if missing:
    print(f'MISSING IDs: {missing}')
else:
    print('All IDs found in HTML')

# Also check for querySelector issues
qsels = re.findall(r"querySelector\(['\"]([^'\"]+)['\"]\)", script)
print(f'\nquerySelector calls: {len(set(qsels))}')

# Check for template literals with backticks that might cause issues
tpl = re.findall(r'`[^`]*`', script)
print(f'Template literals: {len(tpl)}')
