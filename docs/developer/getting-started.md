# Getting started

## Setting up

We use `pixi` as a dependency and dev-tool manager.<br>
Install pixi following their documentation: [pixi installation](https://pixi.prefix.dev/dev/installation/).

### Dependencies

mcstastox is a library, so the package dependencies are never pinned.
Lower bounds are fine and individual versions can be excluded.
See, e.g., [Should You Use Upper Bound Version Constraints](https://iscinumpy.dev/post/bound-version-constraints/) for an explanation.

Development dependencies (as opposed to dependencies of the deployed package that users need to install) are pinned to an exact version in order to ensure reproducibility.
This also includes dependencies used for the various CI builds.
This is done by locking dependencies using pixi, `pixi lock` or `pixi install`.
Specific version constraints can be configured in `pixi.toml`.

Development dependencies are specified in `pixi.lock` and can be installed using

```sh
pixi install --locked --all
```

Additionally, building the documentation requires [pandoc](https://pandoc.org/) which is not on PyPI is listed in the `pixi.toml` as a conda dependency.

### Set up git hooks

The CI pipeline runs a number of code formatting and static analysis tools.
If they fail, a build is rejected.
To avoid that, you can run the same tools locally.
This can be done conveniently using [pre-commit](https://pre-commit.com/) or [prek](https://prek.j178.dev/):

```sh
pre-commit install
```

Alternatively, if you want a different workflow, take a look at ``pixi.toml`` or ``.pre-commit.yaml`` to see what tools are run and how.

## Running unit tests
```sh
pixi run test
```

## Building the docs

`````{tab-set}
````{tab-item} pixi
Build the documentation

```sh
pixi run docs
```

Link check of the built documentation
```sh
pixi run linkcheck
```
````

````{tab-item} Manually

Build the documentation using

```sh
pixi run -e docs python -m sphinx -v -b html -d build/docs_doctrees docs build/html
```

Additionally, test the documentation using

```sh
pixi run -e docs python -m sphinx -v -b doctest -d .tox/docs_doctrees docs html
```
````
`````

## Pixi Quick Guide

### Lock dependencies.

```bash
pixi lock --all
```

### Install virtual environment.

```bash
pixi install --all  # use -e {env-name} flag instead of --all if you want to install specific environment.
# or
pixi reinstall --all
```

If the `pixi.lock` file is up-to-date, you can skip the dependency resolving with `--locked` flag.
```bash
pixi install --locked --all
```

### Activate virtual environment in terminal.

`pixi shell` is similar to `conda activate`.

If you want to activate non-default environment, you can specify which environment you want to activate.

`pixi shell -e lint`

`exit` command deactivates the virtual environment.

You can also simply use `pixi run` to run a command in the virtual environment.<br>
For example, `pixi run nvim .` starts neo-vim in the `default` environment.<br>
Specifying specific environment is also possible, <br>
e.g. to run pre-commit hook for specific file.
```bash
pixi run -e lint pre-commit run --files src/LoadFile.py
```

### Tasks.

Many development routines such as running tests or building docs are defined as `tasks` in the `pixi.toml` file.

To list the available tasks:
```
pixi task list
```

Then you can run the task using `pixi run` command.

e.g. To run unit tests:
```
pixi run test
```

To run static analysis:
```
pixi run check
```

You can pass extra argument to the underlying command wrapped in tasks using `--` indicator.

e.g. if you want to only run the `tests/mcstas_config_test.py`

```
pixi run test -- tests/mcstas_config_test.py
```

