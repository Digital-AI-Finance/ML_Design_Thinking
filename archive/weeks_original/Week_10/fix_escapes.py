with open('part5_practice.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all patterns where we have \\n in the LaTeX source
# These appear as literal backslash-backslash-n in the file
content = content.replace('\\\\n', '\\')

with open('part5_practice.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed all escape sequences')