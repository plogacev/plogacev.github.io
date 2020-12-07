#!/bin/bash
for N in [^_]*.Rmd; do Rscript -e 'rmarkdown::render("'$N'")'; done

