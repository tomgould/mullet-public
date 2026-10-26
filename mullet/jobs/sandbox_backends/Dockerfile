# The image Mullet's docker and podman sandbox backends run job commands in.
# Build it with `mullet sandbox build-image`, which tags it as the
# `sandbox.image` setting says (default mullet-sandbox:<Mullet version>) and
# sets MULLET_VERSION.
#
# It holds a small userland: a shell and coreutils, Python 3 (which also runs
# Mullet's own PID 1 in every container), git, Node.js and npm. Mullet runs it
# as your own uid with a read-only root, no capabilities and no network, so
# there is no USER, ENTRYPOINT or CMD here. A repo that needs other toolchains
# gets its own image (FROM this one, which keeps the labels check() needs)
# named in `sandbox.image`.
#
# The base is pinned by digest: the multi-arch index of debian:trixie-slim as
# of 25 October 2026. To move it on, look up the current one with
# `docker buildx imagetools inspect debian:trixie-slim` and replace it here.
FROM debian:trixie-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad

ARG MULLET_VERSION
LABEL org.opencontainers.image.title="mullet-sandbox" \
      org.opencontainers.image.description="Userland for Mullet's container sandbox backend" \
      mullet.sandbox.version="${MULLET_VERSION}"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates git make nodejs npm procps python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*
