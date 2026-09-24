#!/usr/bin/env sh
# Start the review console and open it in the browser. Linux / macOS
# counterpart of serve.bat; see there for why this exists.
#
# The vault key, in order: DEIDKIT_KEY_URI or DEIDKIT_VAULT_KEY if already
# set (production), otherwise the development key in out/dev-vault.key with a
# warning, otherwise stop. Extra arguments pass through: ./serve.sh --port 9000
set -eu
cd "$(dirname "$0")"

if [ ! -x .venv/bin/deidkit ]; then
    echo "deidkit is not installed in this folder yet. Run ./setup.sh first." >&2
    exit 1
fi

if [ -z "${DEIDKIT_KEY_URI:-}" ] && [ -z "${DEIDKIT_VAULT_KEY:-}" ]; then
    if [ -f out/dev-vault.key ]; then
        DEIDKIT_VAULT_KEY=$(tr -d '\r\n' < out/dev-vault.key)
        export DEIDKIT_VAULT_KEY
        echo "Using the DEVELOPMENT vault key in out/dev-vault.key."
        echo "  It is stored beside the vault it opens. Fine for synthetic data;"
        echo "  for real data the key belongs in a key service (DEIDKIT_KEY_URI)."
        echo
    else
        echo "No vault key: DEIDKIT_KEY_URI and DEIDKIT_VAULT_KEY are unset and" >&2
        echo "there is no out/dev-vault.key. Ask whoever runs this server." >&2
        exit 1
    fi
fi

echo "Starting the console. Keep this terminal open while you work."
exec .venv/bin/deidkit serve --open "$@"
