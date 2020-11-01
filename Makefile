
all: update_r
	git add *
	git commit -a -m update
	git push


update_r:
	cp -r docs/lecture_notes/R_source/* docs/lecture_notes/R/
