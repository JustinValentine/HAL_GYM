from setuptools import setup

setup(
    name="HAL_Gymnasium",
    version="0.0.1",
    install_requires=[
        "gymnasium==0.29.1",  # This adds gymnasium version 0.28.1 with all extra requirements
        "mujoco==2.2.0",          # Adds mujoco version 2.2.0
    ]
)