#!/bin/bash
#!/bin/bash
cp -r docs/lecture_notes/R_source/* docs/lecture_notes/R/
cp -r docs/lecture_notes/stats_source/* docs/lecture_notes/stats/
git add *
git commit -a -m update
git push
