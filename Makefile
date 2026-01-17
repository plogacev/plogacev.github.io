.PHONY: all render clean preview

all: render

render:
	quarto render

preview:
	quarto preview

clean:
	rm -rf docs/*
