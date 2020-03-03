#all: main.pdf
all: main.pdf main-onecolumn.pdf cover_letter.pdf

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


response_letter.pdf: response_letter.tex
	pdflatex response_letter.tex

cover_letter.pdf: cover_letter.tex
	pdflatex cover_letter.tex

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
main-onecolumn.pdf \
cover_letter.aux \
cover_letter.log \
cover_letter.out \
cover_letter.pdf \
response_letter.aux \
response_letter.log \
response_letter.out \
response_letter.pdf
