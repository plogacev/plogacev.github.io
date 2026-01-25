all: ts_1_synthetic_data.md ts_2_analysis.md

ts_1_synthetic_data.md: ts_1_synthetic_data.ipynb 
	jupyter nbconvert --to markdown --execute ts_1_synthetic_data.ipynb
	jupytext --from md --to myst ts_1_synthetic_data.md

ts_2_analysis.md: ts_2_analysis.ipynb 
	jupyter nbconvert --to markdown --execute ts_2_analysis.ipynb
	jupytext --from md --to myst ts_2_analysis.md
