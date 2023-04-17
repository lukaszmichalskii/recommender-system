VENV=venv
PYTHON=$(VENV)/bin/python3
FILES=$(shell git ls-files '*.py')
CWD=$(shell pwd)

.PHONY: test lint format run clean

build: requirements.txt
	@if [ ! -d $(VENV) ]; then virtualenv -p python3 $(VENV); fi
	@$(PYTHON) -m pip install -r requirements.txt;

run:
	@$(PYTHON) $(CWD)/src/app_runner.py -v -r $(CWD)/test/resources/ratings.csv

format:
	@$(PYTHON) -m black .

test:
	@$(PYTHON) -m unittest discover .

lint:
	@$(PYTHON) -m black --diff --check $(FILES)
	@$(PYTHON) -m pylint --disable=all --enable=unused-import $(FILES)

clean:
	@rm -rf results/ execution.log
