#!/usr/bin/env bash
# Installs the PO-token provider for yt-dlp (bgutil-ytdlp-pot-provider, script mode), pinned.
#
# YouTube flags servers whose requests lack a "proof of origin" token ("Sign in to confirm you're not a bot").
# The provider mints one with YouTube's own BotGuard check, run by Node.js in a sandbox (deploy/node-sandboxed).
# This installs the yt-dlp plugin (pip) and builds the token script, both the same pinned version: a mismatch
# breaks them, so the weekly extractor upgrade leaves them alone. To upgrade, set VERSION/COMMIT and rerun.
#
# The npm build runs install scripts (canvas fetches a native module), so it runs in bwrap too: network on,
# only the build directory writable, no home or repo. Only the built output is kept: without src/ the
# plugin's deno variant is unavailable, which would otherwise be preferred and run outside the sandbox.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=2.0.1
COMMIT=2df09aeaa71a4eec1e31901e84fbddbf0c7c54a9  # tag 2.0.1; tags can be moved, the commit can't
dest=data/bgutil/server

.venv/bin/pip install --quiet --disable-pip-version-check "bgutil-ytdlp-pot-provider==$VERSION"

build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$VERSION" https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git "$build/repo"
got=$(git -C "$build/repo" rev-parse HEAD)
if [ "$got" != "$COMMIT" ]; then
  echo "tag $VERSION points to $got, expected $COMMIT: refusing to build" >&2
  exit 1
fi

bwrap --unshare-all --share-net --die-with-parent \
  --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
  --ro-bind-try /etc/alternatives /etc/alternatives \
  --ro-bind-try /etc/resolv.conf /etc/resolv.conf --ro-bind-try /run/systemd/resolve /run/systemd/resolve \
  --ro-bind-try /etc/hosts /etc/hosts --ro-bind-try /etc/nsswitch.conf /etc/nsswitch.conf \
  --ro-bind-try /etc/ssl /etc/ssl --ro-bind-try /etc/ca-certificates /etc/ca-certificates \
  --proc /proc --dev /dev --tmpfs /tmp \
  --bind "$build/repo/server" /build --chdir /build \
  --clearenv --setenv PATH /usr/bin --setenv HOME /tmp --setenv npm_config_cache /tmp/npm \
  /bin/sh -c 'npm ci --no-audit --no-fund && ./node_modules/.bin/tsc && npm prune --omit=dev --no-audit --no-fund'

rm -rf "$dest.new"
mkdir -p "$dest.new"
cp -a "$build/repo/server/build" "$build/repo/server/node_modules" "$build/repo/server/package.json" "$dest.new/"
rm -rf "$dest"
mv "$dest.new" "$dest"

# Checks: the script answers through the sandbox with the plugin's version, and the deno variant is absent.
test ! -e "$dest/src" || { echo "src/ must not be in $dest" >&2; exit 1; }
export POT_SERVER_HOME=$PWD/$dest POT_CACHE=$PWD/data/bgutil/cache
deploy/node-sandboxed "$PWD/$dest/build/generate_once.js" --version
echo "bgutil-ytdlp-pot-provider $VERSION installed ($COMMIT)"
