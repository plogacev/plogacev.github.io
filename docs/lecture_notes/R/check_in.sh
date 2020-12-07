#!/bin/bash
./generate.sh
git add *
git commit -a -m update
git push
cd website
./check_in.sh
