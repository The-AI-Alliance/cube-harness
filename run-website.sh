#!/usr/bin/env zsh
# Run the Jekyll website locally.

: ${PORT:=4000}
bundle exec jekyll serve --port $PORT --baseurl '' --incremental