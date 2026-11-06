#!/usr/bin/env bash
# Production ratchet: no NEW panic site in a library crate's non-test code.
#
# A decompiler runs on files it did not produce. `unwrap()`, `expect()`, `panic!()`,
# `todo!()` and `unimplemented!()` in the load/decode/discover/lift/demangle path are
# reachable crashes on hostile input, so the count is baselined per crate and may only
# go DOWN. Going down is progress and the gate tells you to lower the baseline in the
# same commit; going up is a red build.
#
# Test code is excluded by the repo's layout convention: unit tests live at the BOTTOM
# of a file behind `#[cfg(test)]`, so everything from that line onward in the same file
# is not production code. `crates/verify/src/hw.rs`'s `Width::W1 => unimplemented!()`
# arms are counted and stay counted: they are loud by construction, and a ratchet that
# exempts them by name is a ratchet that lies.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
BASE=.gate/unwrap-baseline
PATTERN='\.unwrap()[()]|\.expect\(|panic!\(|todo!\(|unimplemented!\('

if [ ! -f "$BASE" ]; then
  echo "RATCHET FAIL: baseline file $BASE is missing (a gate that checks nothing must not pass)"
  exit 1
fi

count_crate() { # non-test panic sites in a crate's src/
  total=0
  for f in $(find "crates/$1/src" -name '*.rs' 2>/dev/null | sort); do
    cut=$(grep -n '^#\[cfg(test)\]' "$f" | head -1 | cut -d: -f1)
    if [ -n "$cut" ]; then
      n=$(head -n $((cut - 1)) "$f" | grep -cE "$PATTERN")
    else
      n=$(grep -cE "$PATTERN" "$f")
    fi
    total=$((total + n))
  done
  echo "$total"
}

fail=0
checked_files=$(find crates/*/src -name '*.rs' | wc -l)
if [ "$checked_files" -eq 0 ]; then
  echo "RATCHET FAIL: found zero source files to check"
  exit 1
fi

for crate in core arch-x86 demangle verify cli; do
  want=$(grep -E "^$crate " "$BASE" | awk '{print $2}')
  if [ -z "$want" ]; then
    echo "RATCHET FAIL: no baseline entry for crate '$crate'"
    fail=1
    continue
  fi
  got=$(count_crate "$crate")
  if [ "$got" -gt "$want" ]; then
    echo "RATCHET FAIL: crates/$crate has $got panic sites in non-test code, baseline is $want"
    for f in $(find "crates/$crate/src" -name '*.rs' | sort); do
      cut=$(grep -n '^#\[cfg(test)\]' "$f" | head -1 | cut -d: -f1)
      if [ -n "$cut" ]; then head -n $((cut - 1)) "$f" | grep -nE "$PATTERN" | sed "s|^|  $f:|";
      else grep -nE "$PATTERN" "$f" | sed "s|^|  $f:|"; fi
    done
    echo "  -> return Err/None with a message naming the input, and count a gap; never panic on a byte you did not produce"
    fail=1
  elif [ "$got" -lt "$want" ]; then
    echo "RATCHET: crates/$crate $want -> $got. Lower the baseline in $BASE in this commit."
    fail=1
  else
    echo "ratchet ok: crates/$crate $got panic sites (= baseline)"
  fi
done
exit $fail
