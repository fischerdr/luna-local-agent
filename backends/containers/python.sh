# Sourced by installer/install.sh when it finds no Python 3.10-3.13.
# Only a newer Python (3.14+), but Podman is here: what needs an older one
# - her voice (Kokoro) and llama.cpp - runs in containers instead, so
# Luna's own Python can be this one. The installer offers the containers.
if command -v podman >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1 &&
   python3 -c 'import sys; sys.exit(not sys.version_info[:2] > (3, 13))'; then
    echo "Python $(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])') is newer than her voice supports:" \
         "it and her model server can run in Podman containers instead."
    PY=python3
fi
