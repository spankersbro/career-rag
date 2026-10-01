#!/bin/sh
set -e
git config core.hooksPath .githooks
chmod +x .githooks/*
echo "Hooks installed. Put private terms (names, companies) in data/pii/denylist.txt; it never leaves this machine."
