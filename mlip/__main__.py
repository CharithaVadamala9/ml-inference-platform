"""Enable `python -m mlip ...` as the canonical way to run the CLI.

This is preferred over the installed `mlip` console script during local
development: it resolves the package from the working tree, which sidesteps a
uv editable-install quirk in some versions where the generated `.pth` isn't
honored at interpreter startup. The `mlip` console script still works for
non-editable (wheel) installs.
"""

from mlip.cli import app

if __name__ == "__main__":
    app()
