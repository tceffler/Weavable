all: pdf

pdf:
	doxygen
	cd latex; pdflatex refman
	cd latex; pdflatex refman

clean:
	cd latex && $(MAKE) clean
	cd html && $(MAKE) clean
