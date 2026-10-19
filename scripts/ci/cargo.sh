#!/usr/bin/env bash
# Run `cargo "$@"` (e.g. `scripts/ci/cargo.sh test --all-targets`) and, on
# failure, surface failing tests, panic messages and compile/build-script
# errors as GitHub Actions error annotations, which are visible in the PR UI
# and through the public check-runs API (unlike raw job logs).
set -o pipefail
log=$(mktemp)
cargo "$@" 2>&1 | tee "$log"
status=$?
if [ "$status" -ne 0 ]; then
  grep -E "^test .* FAILED$" "$log" | sed -E 's/^test (.*) \.\.\. FAILED$/::error title=test failed::\1/' | head -50
  grep -E "panicked at" -A2 "$log" | grep -vE "^--$" | head -60 | tr '\n' ' ' | fold -w 900 | sed 's/^/::error title=panic::/'
  grep -E "^error(\[E[0-9]+\])?:|failed to run custom build command|--- stderr" -A12 "$log" | head -80 | tr '\n' ' ' | fold -w 900 | sed 's/^/::error title=build::/'
fi
exit "$status"
