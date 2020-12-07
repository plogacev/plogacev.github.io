#!/bin/bash
rm -r docs/lecture_notes/R/*
rm -r docs/lecture_notes/stats/*
cp -r docs/lecture_notes/R_source/* docs/lecture_notes/R/
cp -r docs/lecture_notes/stats_source/* docs/lecture_notes/stats/
git add *
git commit -a -m update
git push
