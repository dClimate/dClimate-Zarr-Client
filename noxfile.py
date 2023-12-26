import pathlib

import nox

# aiohttp is not building on Python 3.12
ALL_INTERPRETERS = (
    "3.10",
    "3.11",
)
CODE = "dclimate_zarr_client"
DEFAULT_INTERPRETER = "3.10"
HERE = pathlib.Path(__file__).parent


@nox.session(py=ALL_INTERPRETERS)
def tests(session):
    session.install("-e", ".[testing]")
    session.run(
        "pytest",
        f"--cov={CODE}",
        "--cov=tests",
        "--cov-append",
        "--cov-config",
        HERE / ".coveragerc",
        "--cov-report=term-missing",
        "tests",
    )


@nox.session(py=DEFAULT_INTERPRETER)
def cover(session):
    session.install("coverage")
    session.run("coverage", "report", "--fail-under=92", "--show-missing")
    session.run("coverage", "erase")


@nox.session(py=DEFAULT_INTERPRETER)
def lint(session):
    session.install("black", "flake8", "flake8-pyproject")
    run_black(session, check=True)
    session.run("flake8", CODE, "tests")


@nox.session(py=DEFAULT_INTERPRETER)
def blacken(session):
    # Install all dependencies.
    session.install("black")
    run_black(session)


def run_black(session, check=False):
    args = ["black"]
    if check:
        args.append("--check")
    args.extend(["noxfile.py", CODE, "tests"])
    session.run(*args)
