from setuptools import setup, find_packages

setup(
    name="sn_line_vel",
    version="0.1",
    packages=find_packages(),  # Automatically find all packages and subpackages
    # package_data={"bayespecfit": [""]},
    include_package_data=True,
    install_requires=["pymc", "corner"],  # Add dependencies here
)
