#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

git submodule update --init cap/third_party/f1tenth_gym

uv python install 3.10
if [[ ! -x .venv-f1tenth/bin/python ]]; then
  uv venv .venv-f1tenth --python 3.10
fi

# The upstream package metadata still pins Gym 0.19 and NumPy <=1.22. The
# simulator source is compatible with ASPIRE's tested Python 3.10 stack, so
# install a pinned compatibility set and the simulator source itself without
# resolving those stale constraints.
source .venv-f1tenth/bin/activate
uv sync --locked --active --extra dev --inexact
uv pip install --python .venv-f1tenth/bin/python \
  "gym==0.25.2" \
  "numba==0.61.2" \
  "pyglet==2.1.15" \
  "PyOpenGL==3.1.6"
uv pip install --python .venv-f1tenth/bin/python --no-deps -e cap/third_party/f1tenth_gym
deactivate

.venv-f1tenth/bin/python -c \
  "import f110_gym, gym, numba, numpy; print('f1tenth suite ready:', gym.__version__, numba.__version__, numpy.__version__)"
