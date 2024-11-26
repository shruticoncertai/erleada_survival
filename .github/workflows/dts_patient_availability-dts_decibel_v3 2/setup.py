from pathlib import Path
import setuptools
from sys import path

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as rf:
    requirements = rf.read()
    requirements = requirements.split('\n')
path.append(str(Path(__file__).parent.resolve()))
setuptools.setup(
    name="dts_patient_availability",
    version="0.5",
    author="Jaina",
    author_email="jbharath@concertai.com",
    description="Patient availability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PrecisionHealthIntelligence/dts_patient_availability",
    package_dir={'': 'src/'},
    packages=setuptools.find_packages(where='src/'),
    python_requires='>=3.6',
    classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    include_package_data=True,
    install_requires=requirements
)
print('Packages Found : ', setuptools.find_packages())
