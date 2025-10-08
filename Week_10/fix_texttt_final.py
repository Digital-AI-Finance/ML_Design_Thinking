with open('part5_practice.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# In multi-line \texttt blocks, we don't need \\ at all
# The \\ is causing "There's no line here to end" errors
# \texttt{} handles line breaks automatically

# Replace \\ with a space inside texttt blocks (for readability)
import re

def fix_texttt_linebreaks(content):
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
            if content[j] == '\\' and j+1 < len(content) and content[j+1] == '{':
                # Escaped brace doesn't count
                j += 2
                continue
            if content[j] == '{':
                brace_count += 1
            elif content[j] == '}':
                brace_count -= 1
            j += 1

        # Extract texttt content
        texttt_content = content[start+8:j-1]

        # Replace \\ with actual newline for proper formatting
        fixed_content = texttt_content.replace('\\\\', '\\newline\n')

        result.append(fixed_content)
        result.append('}')

        i = j

    return ''.join(result)

fixed_content = fix_texttt_linebreaks(content)

with open('part5_practice.tex', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print('Fixed linebreaks in texttt blocks')