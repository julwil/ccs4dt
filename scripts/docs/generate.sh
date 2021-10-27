#!/usr/bin/env bash

echo "Generating documentation"
sphinx-apidoc -o docs . ccs4dt/tests -f
cd docs
make html
cp _build/html/* . -rf
rm _build -rf