##
# LLM Meta-analysis of scientific literature
#
# @file
# @version 0.1
#
SHELL:=/bin/bash
PROJECT=funk-etal-2008-meta
PYTHON_VERSION=3.11.7
VENV_NAME=${PROJECT}-${PYTHON_VERSION}
VENV_DIR=$(shell pyenv root)/versions/${VENV_NAME}
VENV_BIN=${VENV_DIR}/bin
PYTHON=${VENV_BIN}/python
JUPYTER_ENV_NAME="python (${VENV_NAME})"

## Make sure you have `pyenv` and `pyenv-virtualenv` installed beforehand
##
## https://github.com/pyenv/pyenv
## https://github.com/pyenv/pyenv-virtualenv
##
## On a Mac: $ brew install pyenv pyenv-virtualenv
##
## Configure your shell via:
##   https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv
##

# .ONESHELL:
DEFAULT_GOAL: help
.PHONY: help run clean build venv ipykernel update jupyter

# Colors for echos
ccend=$(shell tput sgr0)
ccbold=$(shell tput bold)
ccgreen=$(shell tput setaf 2)
ccso=$(shell tput smso)

clean: ## >> remove all environment and build files
	@echo ""
	@echo "$(ccso)--> Removing virtual environment $(ccend)"
	pyenv virtualenv-delete --force ${VENV_NAME}
	rm .python-version

env: ##@main >> build the virtual environment with an ipykernel for jupyter and install requirements
	@echo ""
	@echo "$(ccso)--> Build $(ccend)"
	$(MAKE) install
	$(MAKE) ipykernel

venv: $(VENV_DIR) ## >> set up the virtual environment

$(VENV_DIR):
	@echo "$(ccso)--> Create pyenv virtualenv $(ccend)"
	pyenv install -s $(PYTHON_VERSION)
	pyenv virtualenv ${PYTHON_VERSION} ${VENV_NAME}
	echo ${VENV_NAME} > .python-version

requirements.txt: requirements.in
	$(PYTHON) -m piptools compile --upgrade --resolver=backtracking requirements.in -o requirements.txt


install: venv requirements.txt ##@main >> install tenforty and deps into venv
	@echo "$(ccso)--> Updating packages $(ccend)"
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PYTHON) -m pip install -U pip-tools
	$(PYTHON) -m pip install -r requirements.txt

ipykernel: venv ##@main >> install a Jupyter iPython kernel using our virtual environment
	@echo ""
	@echo "$(ccso)--> Install ipykernel to be used by jupyter notebooks $(ccend)"
	$(PYTHON) -m pip install ipykernel jupyterlab watermark jupyter_black
    # Until nb-black's upstream is fixed...
	$(PYTHON) -m pip install git+https://github.com/IsaGrue/nb_black.git
	$(PYTHON) -m ipykernel install --user --name=$(VENV_NAME) --display-name=$(JUPYTER_ENV_NAME)


# Other commands.
hooks: ##@options >> install pre-commit hooks
	pre-commit install

update-hooks: ##@options >> bump all hooks to latest versions
	pre-commit autoupdate

run-hooks: ##@options >> run hooks over staged files
	pre-commit run

run-hooks-all-files: ##@options >> run hooks over ALL files in workspace
	pre-commit run -a


## This help screen
help:
	@printf "Available targets:\n\n"
	@awk '/^[a-zA-Z\-_0-9%:\\]+/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
		helpCommand = $$1; \
		helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
	gsub("\\\\", "", helpCommand); \
	gsub(":+$$", "", helpCommand); \
		printf "  \x1b[32;01m%-35s\x1b[0m %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST) | sort -u
	@printf "\n"
	@printf "To activate the environment in your local shell type:\n"
	@printf "   $$ pyenv activate $(VENV_NAME)\n"
