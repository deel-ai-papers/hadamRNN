.PHONY: help prepare-dev test test-disable-gpu doc serve-doc
.DEFAULT: help

help:
	@echo "make prepare-dev"
	@echo "       create and prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests and linting on py36, py37, py38"
	@echo "make check_all"
	@echo "       check all files using pre-commit tool"
	@echo "make updatetools"
	@echo "       updatetools pre-commit tool"
	@echo "make test-disable-gpu"
	@echo "       run test with gpu disabled"
	@echo "make serve-doc"
	@echo "       run documentation server for development"
	@echo "make doc"
	@echo "       build mkdocs documentation"

prepare-dev:
	python3 -m pip install virtualenv
	python3 -m venv <lib>_dev_env
	. <lib>_dev_env/bin/activate && pip install --upgrade pip
	. <lib>_dev_env/bin/activate && pip install -r requirements.txt
	. <lib>_dev_env/bin/activate && pip install -r requirements_dev.txt
# or . <lib>_dev_env/bin/activate && pip install -e .[dev] #install_requires and dev in extras_require (setup.py)
	. <lib>_dev_env/bin/activate && pre-commit install
	. <lib>_dev_env/bin/activate && pre-commit install-hooks
	. <lib>_dev_env/bin/activate && pre-commit install --hook-type commit-msg

test:
	. <lib>_dev_env/bin/activate && tox

check_all:
	. <lib>_dev_env/bin/activate && pre-commit run --all-files

updatetools:
	. <lib>_dev_env/bin/activate && pre-commit autoupdate

test-disable-gpu:
	. <lib>_dev_env/bin/activate && CUDA_VISIBLE_DEVICES=-1 tox

doc:
	. <lib>_dev_env/bin/activate && mkdocs build
	. <lib>_dev_env/bin/activate && mkdocs gh-deploy

serve-doc:
	. <lib>_dev_env/bin/activate && CUDA_VISIBLE_DEVICES=-1 mkdocs serve
