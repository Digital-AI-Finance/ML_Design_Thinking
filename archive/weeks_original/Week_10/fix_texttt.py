import re

with open('part5_practice.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all \texttt{...} blocks and replace \\\\ with \\
# Within \texttt, \\\\ means "execute line break twice" which is an error
# We want just \\ for a visual line break in monospace

def fix_texttt(match):
    texttt_content = match.group(1)
    # Replace \\\\ with \\ inside texttt blocks
    fixed_content = texttt_content.replace('\\\\\\\\', '\\\\')
    return f'\\texttt{{{fixed_content}}}'

# Match \texttt{...} blocks (non-greedy)
content = re.sub(r'\\texttt\{(.*?)\}', fix_texttt, content, flags=re.DOTALL)

with open('part5_practice.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed \\\\  to \\ in texttt blocks')