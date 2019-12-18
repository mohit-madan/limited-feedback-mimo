#all: main.pdf
all: main.pdf main-onecolumn.pdf

MAINFILES := main.tex

IMAGEFILES := images/new-adpm.svg

IMAGEPDFS := $(IMAGEFILES:%.svg=%.pdf)

images/new-adpm.pdf: images/new-adpm.svg
	inkscape --export-pdf=images/new-adpm.pdf images/new-adpm.svg

main.pdf: main.bib $(MAINFILES) $(IMAGEPDFS)
	pdflatex main.tex && bibtex main && pdflatex main && pdflatex main

main-onecolumn.pdf: main.bib $(MAINFILES) $(IMAGEPDFS)
	sed '1s/.*/\\documentclass[journal,12pt,onecolumn,draftcls]{IEEEtran}/' main.tex > main-onecolumn.tex
	pdflatex main-onecolumn.tex && bibtex main-onecolumn && pdflatex main-onecolumn && pdflatex main-onecolumn
	$(RM) main-onecolumn.tex


.PHONY: clean
clean:
	$(RM) $(IMAGEPDFS) main.aux \
main.bbl \
main.blg \
main.log \
main.out \
main.pdf \
main-onecolumn.aux \
main-onecolumn.bbl \
main-onecolumn.blg \
main-onecolumn.log \
main-onecolumn.out \
main-onecolumn.pdf