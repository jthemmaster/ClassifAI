# Makefile
SHELL = /bin/bash

# Styling
.PHONY: style
style:
	black --line-length 79 .
	flake8 --exclude=venv
	python3 -m isort .
	pyupgrade

.PHONY: clean
clean:
	rm -rf build dist *.egg-info
