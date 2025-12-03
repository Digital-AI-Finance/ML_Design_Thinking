with open('part5_practice.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# In \texttt{}, we should use \\ for line breaks, not \\\\
# The pattern \\\\  is causing "There's no line here to end" errors

# Replace \\\\ with \\ but only inside \texttt{} blocks
import re

def fix_inside_texttt(content):
    result = []
    i = 0
    while i < len(content):
        # Find next \texttt{
        start = content.find('\\texttt{', i)
        if start == -1:
            # No more texttt blocks
            result.append(content[i:])
            break

        # Add everything before \texttt{
        result.append(content[i:start+8])  # include \texttt{

        # Find matching }
        brace_count = 1
        j = start + 8
        while j < len(content) and brace_count > 0:
            if content[j] == '{':
                brace_count += 1
            elif content[j] == '}':
                brace_count -= 1
            j += 1

        # Extract texttt content
        texttt_content = content[start+8:j-1]

        # Fix \\\\  to \\ in this block
        fixed_content = texttt_content.replace('\\\\\\\\', '\\\\')

        result.append(fixed_content)
        result.append('}')

        i = j

    return ''.join(result)

fixed_content = fix_inside_texttt(content)

with open('part5_practice.tex', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print('Fixed double backslashes in texttt blocks')