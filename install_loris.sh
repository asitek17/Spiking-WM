#!/usr/bin/env bash
# Custom workaround for loris 0.5.3 installation failure with modern setuptools.
#
# Root cause: setup.py contains the line
#   __builtins__.__NUMPY_SETUP__ = False
# which sets an attribute on __builtins__. When setuptools runs setup.py via exec(),
# __builtins__ is a plain dict rather than a module, so attribute assignment raises
#   AttributeError: 'dict' object has no attribute '__NUMPY_SETUP__'
#
# Additionally, loris uses C++ numpy internals removed in numpy 2.0, so numpy must
# be pinned to 1.26.x before building. numpy==1.26.4 is installed first in the
# Dockerfile (via requirements.txt) before this script runs.
#
# Via uv: called automatically from setup.sh after `uv sync` (numpy 1.26.4 already present).
# Manually: activate the venv first, then run:
#   source .venv/bin/activate && bash install_loris.sh
set -e

ARCHIVE=/tmp/loris-0.5.3.tar.gz
SRC_DIR=/tmp/loris-0.5.3

# Download loris 0.5.3 source archive
curl -L https://files.pythonhosted.org/packages/source/l/loris/loris-0.5.3.tar.gz \
    -o "$ARCHIVE"

# Extract the archive
tar -xzf "$ARCHIVE" -C /tmp

# Remove the line incompatible with modern setuptools
sed -i '/__builtins__\.__NUMPY_SETUP__ = False/d' "$SRC_DIR/setup.py"

# Install without build isolation so the already-pinned numpy (1.26.4) is used
# for C++ extension compilation — loris uses deprecated numpy 1.x internal APIs
uv pip install "$SRC_DIR" --no-build-isolation

echo "loris installed successfully"
