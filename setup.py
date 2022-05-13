import os
from typing import List
import re
from importlib.util import module_from_spec, spec_from_file_location

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")


def _load_requirements(
        path_dir: str,
        file_name: str = "base.txt",
        comment_char: str = "#",
        unfreeze: bool = True
) -> List[str]:
    """Load requirements from a file.

    >>> _load_requirements(os.path.join("root_path", "requirements"))
    ['numpy...', 'tensorflow...', ...]
    """
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [line.strip() for line in file.readlines()]
    requirements = []
    for line in lines:
        if comment_char in line:
            line = line[: line.index(comment_char)]
        comment = line[line.index(comment_char):] if comment_char in line else ""
        req = line.strip()

        # skip directly installed dependencies
        if not req or req.startswith("http") or "@http" in req:
            continue
        # remove version restrictions unless they are strict
        if unfreeze and "<" in req and "strict" not in comment:
            req = re.sub(r",? *<=? *[\d\.\*]+", "", req).strip()
        requirements.append(req)

    return requirements


# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
# Define package extras. These are only installed if you specify them.
# From remote, use like `pip install recsys[dev, tf]`
# From local copy of repo, use like `pip install ".[dev, tf]"`
extras = {
    "tf": _load_requirements(path_dir=_PATH_REQUIRE, file_name="tf.txt"),
    "tf_gpu": _load_requirements(path_dir=_PATH_REQUIRE, file_name="tf_gpu.txt"),
    "scann": _load_requirements(path_dir=_PATH_REQUIRE, file_name="scann.txt"),
    "extra": _load_requirements(path_dir=_PATH_REQUIRE, file_name="extra.txt")
}

extras["all_cpu"] = extras["tf"] + extras["extra"]
extras["all_gpu"] = extras["tf_gpu"] + extras["extra"]

setup(
    name='recsys',
    version='0.1',
    description='Recommendation system',
    long_description='',
    long_description_content_type="text/markdown",
    author='Analytic Factory',
    author_email='analytic.factory@pedidosya.com',
    packages=find_packages(),
    install_requires=_load_requirements(_PATH_REQUIRE),
)
