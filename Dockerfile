# deidkit -- de-identification pipeline for inbound EDC data
#
# Two stages so the runtime image carries no build toolchain. The runtime user
# is non-root and owns nothing writable except the paths mounted into it: a
# process that handles PHI should not be able to modify its own code.
#
# Build:
#   docker build -t deidkit:0.1.0 .
#   docker build -t deidkit:0.1.0-full --build-arg EXTRAS='[all]' \
#                --build-arg SPACY_MODEL=en_core_web_lg .
#
# The default build is core-only: small, fast, and the free-text screen falls
# back to the built-in pattern detector. Pass EXTRAS to include Presidio, the
# SAS readers, cloud storage backends and Parquet.

# ----------------------------------------------------------------------
FROM python:3.12-slim AS build

ARG EXTRAS=""
ARG SPACY_MODEL=""

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# pyreadstat builds from source on some platforms; give it a compiler here so
# the runtime stage never needs one.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install ".${EXTRAS}"

# The spaCy model is ~560 MB, so it is opt-in. Without it Presidio cannot
# start and the pipeline falls back to the pattern detector -- deliberately,
# but silently, which is why the runtime prints which detector is live.
RUN if [ -n "$SPACY_MODEL" ]; then \
        /opt/venv/bin/python -m spacy download "$SPACY_MODEL"; \
    fi

# ----------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="deidkit" \
      org.opencontainers.image.description="De-identification pipeline for inbound EDC/SDTM clinical data" \
      org.opencontainers.image.licenses="Proprietary"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Production default: refuse to fall back to a bare environment variable
    # for the vault key. Override only in a development compose file.
    DEIDKIT_REQUIRE_MANAGED_KEY=1

COPY --from=build /opt/venv /opt/venv
COPY scripts/ /app/scripts/
COPY contracts/ /app/contracts/

# Non-root, and the app tree stays read-only to it. The three data zones are
# mount points, not image contents -- see compose.yaml and DEPLOY.md.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin deid \
 && mkdir -p /data/quarantine /data/tiers /vault \
 && chown -R deid:deid /data /vault \
 && chmod -R a-w /app /opt/venv

USER deid
WORKDIR /app

# Fails the build if the package cannot import, so a broken image never ships.
RUN python -c "import deidkit; print('deidkit', deidkit.__version__)"

HEALTHCHECK --interval=60s --timeout=10s --retries=2 \
    CMD python -c "import deidkit, sys; sys.exit(0)"

ENTRYPOINT ["deidkit"]
CMD ["--help"]
