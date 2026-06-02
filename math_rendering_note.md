# Math Rendering Note

Some mathematical formulas in `README.md` and `replication_note.md` are written in LaTeX-style Markdown math notation.

For example:

```latex
$$
Y_{t_i}
\approx
\sum_{j=0}^{i}
\exp\!\left[-\beta\left(t_i-t_j\right)\right]
\lambda u_{t_j}\Delta t .
$$
```

These formulas are intended to look similar to the notation used in the original paper, *Solving Optimal Execution Problems via In-Context Operator Networks*.

However, GitHub's Markdown preview does not always support every LaTeX command or rendering pattern. As a result, some formulas may not display correctly on the GitHub webpage.

If the formulas do not render properly on GitHub, please download or clone the repository and open the Markdown files in VSCode.

Recommended viewing method:

1. Open the project folder in VSCode.
2. Open `README.md` or `replication_note.md`.
3. Press `Cmd + Shift + V` on macOS to open Markdown Preview.

The formulas should render more clearly in VSCode's Markdown preview than in GitHub's web preview.
