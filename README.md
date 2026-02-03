# Digital Garden

## Setup

1. Install uv (if needed): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Create venv: `uv venv --python-preference only-managed`
3. Activate: `source .venv/bin/activate`
4. Install dependencies: `uv sync`

## Running

```bash
mkdocs serve    # Dev server at http://127.0.0.1:8000 (live reload)
mkdocs build    # Build static site to site/ directory
```

## Math notes

To do multi-lined equations properly, use the `\aligned` environment, like this: 

$$
\begin{aligned}
H(X,Y) &= -\sum\sum p(x,y)\log p(x,y) \\
&= -\sum\sum p(x,y)\log p(x)p(y|x) \\
&= -\sum\sum p(x,y)\log p(x) - \sum\sum p(x,y)\log p(y|x) \\
&= -\sum p(x)\log p(x) - \sum\sum p(x,y)\log p(y|x) \\
&= H(X) + H(Y|X)
\end{aligned}
$$