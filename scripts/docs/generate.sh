#!/usr/bin/env bash

cd docs
sphinx-apidoc -o . .. ../ccs4dt/tests/
make html
cp _build/html/* . -rf