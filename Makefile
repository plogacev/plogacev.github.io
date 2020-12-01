all:
	make update_r
	make update_stats
	git add *
	git commit -a -m update
	git push


update_r:
	cp -r docs/lecture_notes/R_source/* docs/lecture_notes/R/
update_stats:
	cp -r docs/lecture_notes/stats_source/* docs/lecture_notes/stats/
