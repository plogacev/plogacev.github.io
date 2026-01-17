.PHONY: all render clean preview

all: render
	git add docs/
	git status

render:
	quarto render

preview:
	quarto preview

clean:
	rm -rf docs/*
