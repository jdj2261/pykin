from setuptools import setup, find_packages
print(find_packages())
# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if '.png' not in x]
long_description = ''.join(lines)

setup(
    name="pykin",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "trimesh[easy]",
        "tqdm",
        "pyyaml",
        "python-fcl"
    ],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3',
    description="Robotics Kinematics Library",
    author="Dae Jong Jin",
    url="https://github.com/jdj2261/pykin.git",
	download_url="https://github.com/jdj2261/pykin/archive/refs/heads/main.zip",
    author_email="wlseoeo@gmail.com",
    version="1.1.6",
    long_description=long_description,
    long_description_content_type='text/markdown'
)

# python setup.py sdist bdist_wheel
# twine upload --skip-existing dist/*
