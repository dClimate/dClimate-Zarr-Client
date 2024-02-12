import pathlib

import nox


# 3.9 doesn't work wiht multiformats, which is required for IPFS
S3_ONLY_INTERPRETER = "3.9"

# aiohttp is not building on Python 3.12
IPFS_VALID_INTERPRETERS = (
    "3.10",
    "3.11",
)
CODE = "dclimate_zarr_client"
DEFAULT_INTERPRETER = "3.10"
HERE = pathlib.Path(__file__).parent


@nox.session(py=IPFS_VALID_INTERPRETERS)
def tests_with_ipfs(session):
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


@nox.session(py=S3_ONLY_INTERPRETER)
def test_without_ipfs(session):
    session.install("-e", ".[testing]")
    session.run(
        "pytest", "tests/test_geotemporal_data.py", "tests/test_s3_retrieval.py", "tests/test_zarr_metadata.py"
    )


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
