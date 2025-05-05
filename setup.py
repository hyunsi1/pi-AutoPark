# setup.py
from setuptools import setup, find_packages

setup(
    name="pi_autopark",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "opencv-python",
        "torch",
        "pyyaml",
        "ultralytics",
    ],
)
