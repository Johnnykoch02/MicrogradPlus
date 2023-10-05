from setuptools import setup, find_packages

setup(
    name='MicrogradPlus',
    version='1.0',
    packages=find_packages(),
    url='https://github.com/Johnnykoch02/MicrogradPlus',
    author='Jonathan Koch',
    author_email='johnnykoch02@gmail.com',
    description='A simple NumPy-based automatic differentiation library',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
