"""Find and remove remaining emojis in all project files."""
import os, re

emojis = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
root = os.path.dirname(os.path.abspath(__file__))

count = 0
for dirpath, _, filenames in os.walk(root):
    if '.venv' in dirpath or '__pycache__' in dirpath or '.git' in dirpath:
        continue
    for f in filenames:
        if f.endswith(('.py', '.md', '.html', '.bat', '.json', '.txt', '.yml')):
            path = os.path.join(dirpath, f)
            try:
                content = open(path, 'r', encoding='utf-8').read()
                found = emojis.findall(content)
                if found:
                    rel = os.path.relpath(path, root)
                    print(f"  {rel}: {len(found)} emojis -> removing")
                    cleaned = emojis.sub('', content)
                    open(path, 'w', encoding='utf-8').write(cleaned)
                    count += len(found)
            except Exception:
                pass

print(f"\nTotal emojis removed: {count}")
