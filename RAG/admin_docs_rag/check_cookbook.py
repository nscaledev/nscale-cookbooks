import json, re, sys

SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{20,}|api[_-]?key\s*=\s*['\"][^'\"]{12,})", re.I)
FENCE = re.compile(r"^```python[^\n]*\n(.*?)^```[ \t]*$", re.S | re.M)

def cells_from_md(text):
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)  # drop Jupytext header
    cells, pos = [], 0
    for m in FENCE.finditer(text):
        prose = text[pos:m.start()].strip()
        if prose:
            cells.append({"cell_type": "markdown", "source": prose})
        cells.append({"cell_type": "code", "source": m.group(1).strip()})
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        cells.append({"cell_type": "markdown", "source": tail})
    return cells

def cells_from_ipynb(path):
    nb = json.load(open(path))
    for c in nb["cells"]:
        c["source"] = "".join(c["source"]).strip()
    return nb["cells"]

def check(cells, name, is_nb):
    problems = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if i == 0 or cells[i - 1]["cell_type"] != "markdown":
            problems.append(f"{name} cell {i}: code without prose before it")
        if SECRET.search(c["source"]):
            problems.append(f"{name} cell {i}: possible hardcoded secret")
        if "pip install" in c["source"] and "%%capture" not in c["source"]:
            problems.append(f"{name} cell {i}: pip install without %%capture")
        if is_nb:
            if c.get("execution_count") is None:
                problems.append(f"{name} cell {i}: never executed")
            size = sum(len(json.dumps(o)) for o in c.get("outputs", []))
            if size > 20_000:
                problems.append(f"{name} cell {i}: output is {size} chars; trim it")
    if cells and cells[-1]["cell_type"] != "markdown":
        problems.append(f"{name}: ends on code; add a conclusion")
    return problems

problems, code = [], {}
for path in sys.argv[1:]:
    is_nb = path.endswith(".ipynb")
    cells = cells_from_ipynb(path) if is_nb else cells_from_md(open(path).read())
    problems += check(cells, path, is_nb)
    code[path] = [c["source"] for c in cells if c["cell_type"] == "code"]
if len(code) == 2:
    a, b = code.values()
    if a != b:
        problems.append("code cells differ between .md and .ipynb; run jupytext --sync")

print("\n".join(problems) or "OK")
sys.exit(1 if problems else 0)
