#!/bin/sh
# Render the published decks without access codes and push to gh-pages.
set -e
cd "$(dirname "$0")"
rm -rf _publish
QUARTO_PROFILE=publish quarto render
if [ -f _quarto-class.yml ]; then
  codes=$(sed -n 's/^ *lec[0-9]*: *["'\'']\{0,1\}\([^"'\'' ]\{1,\}\).*/\1/p' _quarto-class.yml)
  if [ -z "$codes" ] && grep -q '^ *lec[0-9]*:' _quarto-class.yml; then
    echo "ERROR: could not read access codes from _quarto-class.yml; not publishing." >&2
    exit 1
  fi
  for code in $codes; do
    if grep -rqF "$code" _publish; then
      echo "ERROR: access code $code found in _publish; not publishing." >&2
      exit 1
    fi
  done
fi

# Push _publish as the sole content of the gh-pages branch.
touch _publish/.nojekyll
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp -R _publish/ "$tmp"
git -C "$tmp" init -q -b gh-pages
git -C "$tmp" add -A
git -C "$tmp" commit -q -m "Publish slides from $(git rev-parse --short HEAD)"
git -C "$tmp" push -q -f "$(git remote get-url origin)" gh-pages
echo "Published to gh-pages."
