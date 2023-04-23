.PHONY: all
all:
	git add .
	git commit -m "updating"
	git push

.PHONY: tex
tex:
	pdflatex meilenstein.tex
	makeglossaries meilenstein.glo
	pdflatex meilenstein.tex
