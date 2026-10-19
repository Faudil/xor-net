#!/usr/bin/env bash
# Run `cargo test "$@"` and, on failure, surface each failing test and its
# panic message as a GitHub Actions error annotation (visible in the PR UI
# and through the public check-runs API, unlike raw job logs).
set -o pipefail
log=$(mktemp)
cargo test "$@" 2>&1 | tee "$log"
status=$?
if [ "$status" -ne 0 ]; then
  grep -E "^test .* FAILED$" "$log" | sed -E 's/^test (.*) \.\.\. FAILED$/::error title=test failed::\1/' | head -50
  grep -E "panicked at" -A2 "$log" | grep -vE "^--$" | head -60 | tr '\n' ' ' | fold -w 900 | sed 's/^/::error title=panic::/'
  grep -E "^error(\[E[0-9]+\])?:" -A3 "$log" | head -40 | tr '\n' ' ' | fold -w 900 | sed 's/^/::error title=build::/'
fi
exit "$status"
