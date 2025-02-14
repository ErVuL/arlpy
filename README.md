# ARL Python Tools modified

A custom version of arlpy with some more tools and propagation models. It uses matplotlib instead of bokeh. This work is a week-end project, it may contains some errors, bugs and mising functionnalities.
Fork from : https://github.com/org-arl/arlpy/

## Usage

### Install

For Debian based distros:

    sudo apt install git texlive-base gfortran cmake
    
For Fedora based distros:

    sudo dnf install git texlive-base gfortran cmake 

Then activate your python virtual env and type:

    cd <installation_path>
    git clone git@github.com:ErVuL/arlpy.git
    git clone git@github.com:ErVuL/pyram.git
    git clone git@github.com:ErVuL/Acoustics-Toolbox.git
    pip install -e ./arlpy
    pip install -e ./pyram
    sudo mkdir -p /opt/build/at
    sudo cp -r Acoustics-Toolbox/* /opt/build/at
    cd /opt/build/at
    sudo make clean
    sudo make all
    sudo make install
    sudo echo 'export PATH="/opt/build/at/bin:$PATH"' >> ~/.bashrc

### Uninstall

Activate your python virtual env and type:

    cd <installation_path>
    pip uninstall arlpy pyram
    rm -rf arlpy
    rm -rf pyram
    rm -rf Acoustics-Toolbox
    sudo rm -rf /opt/build/at
    sed -i '/\/opt\/build\/at\/bin/d' ~/.bashrc
    
You have to manually uninstall dependencies if you want to.

## About

### PYRAM

Range dependant Acoustic Model is a Parabolic Equation solver.\
Python adaptation of RAM v1.5.\
Fork from https://github.com/marcuskd/pyram.

### OALIB AT

OALIB source code (fortran) written in the 80's and 90's. Contains:
  - BELLHOP: Beam/ray trace code
  - KRAKEN: Normal mode code
  - SCOOTER: Finite element FFP code
  - SPARC: Time domain FFP code

Fork from https://github.com/oalib-acoustics/Acoustics-Toolbox/tree/main.

### ARLPY

Python project with some signal processing, underwater acoustics utilities and also able to interact with bellhop.\
Fork from https://github.com/org-arl/arlpy.

### UTM

Bidirectional UTM-WGS84 converter for python.\
Fork from https://github.com/Turbo87/utm.git.
