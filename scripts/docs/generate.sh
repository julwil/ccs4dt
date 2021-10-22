#!/usr/bin/env bash

cd /home/ccs4dt/docs
sphinx-apidoc -o . .. ../ccs4dt/tests/
make html
cp _build/html/* . -rf