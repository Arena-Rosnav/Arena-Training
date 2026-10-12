from setuptools import setup, find_packages

package_name = "arena_training"

# Find all packages including subpackages
packages = find_packages(where=".")

setup(
    name=package_name,
    packages=packages,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    zip_safe=True,
    maintainer="Tuan Anh Roman Le",
    maintainer_email="anh.maxiking@hotmail.de",
    description="The arena_training package for DRL training pipeline",
    license="MIT",
    entry_points={
        "console_scripts": [],
    },
)
