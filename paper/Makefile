#all: main.pdf
all: main.pdf

MAINFILES := main.tex

IMAGEFILES := images/new-adpm.svg

IMAGEPDFS := $(IMAGEFILES:%.svg=%.pdf)

images/new-adpm.pdf: images/new-adpm.svg
	inkscape --export-pdf=images/new-adpm.pdf images/new-adpm.svg

main.pdf: main.bib $(MAINFILES) $(IMAGEPDFS)
	pdflatex main.tex && bibtex main && pdflatex main && pdflatex main

.PHONY: clean
clean:
	$(RM) $(IMAGEPDFS) main.aux \
main.bbl \
main.blg \
main.log \
main.out \
main.pdf
