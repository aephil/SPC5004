#!/bin/sh
# Render the published decks without access codes and push to gh-pages.
set -e
cd "$(dirname "$0")"
rm -rf _site
QUARTO_PROFILE=publish quarto render
if [ -f _quarto-class.yml ]; then
  codes=$(sed -n 's/^ *lec[0-9]*: *["'\'']\{0,1\}\([^"'\'' ]\{1,\}\).*/\1/p' _quarto-class.yml)
  if [ -z "$codes" ] && grep -q '^ *lec[0-9]*:' _quarto-class.yml; then
    echo "ERROR: could not read access codes from _quarto-class.yml; not publishing." >&2
    exit 1
  fi
  for code in $codes; do
    if grep -rqF "$code" _site; then
      echo "ERROR: access code $code found in _site; not publishing." >&2
      exit 1
    fi
  done
fi
quarto publish gh-pages --no-render --no-prompt
