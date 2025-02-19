# ARL Python Tools modified

Python module dedicated to underwater acoustics application.\
Integration of OALIB propagation models and pyram into arlpy. This work is a week-end project, it may contains some errors, bugs and mising functionnalities. I would be happy to receive some pull requests !

Fork from : https://github.com/org-arl/arlpy/

## Usage

The installation process is dedicated to Linux based OS, but an adaptation for Mac or Windows may be possible.

#### Install

Install dependencies:

    sudo apt/dnf install git texlive-base gfortran cmake

Activate your python virtual env, then clone git projects:

    cd <installation_path>
    git clone git@github.com:ErVuL/arlpy.git
    git clone git@github.com:ErVuL/pyram.git
    git clone git@github.com:ErVuL/Acoustics-Toolbox.git

Pip install local modules:

    pip install -e ./arlpy
    pip install -e ./pyram

Install OALIB toolbox:

    sudo mkdir -p /opt/build/at
    sudo cp -r Acoustics-Toolbox/* /opt/build/at
    cd /opt/build/at
    sudo make clean
    sudo make all
    sudo make install
    sudo echo 'export PATH="/opt/build/at/bin:$PATH"' >> ~/.bashrc

If you consider installing the OALIB toolbox somewhere else than the proposed solution, you will have to replace ***</opt/build/at/bin>*** by ***<oalib_installation_path/bin>***  in ***arlpy/uwapm.py***:

    # Add acoustic toolbox path to Python path
    os.environ['PATH'] = os.environ['PATH'].replace(':<oalib_installation_path/bin>', '')+":<oalib_installation_path/bin>"

and in your ***.bashrc***

    export PATH="<oalib_installation_path/bin>:$PATH"

### Examples

Some scripts are available into ***<installation_path>/arlpy/examples***, it contains:

- Power spectral density estimate
- Power spectral density PDF
- Frequency response function estimate
- SEL measurement
- Acoustic propagation modeling (BELLHOP, RAM, KRAKEN)
- Wenz noise modeling

### Uninstall

Activate your python virtual env, then:

    cd <installation_path>
    pip uninstall arlpy pyram
    rm -rf arlpy
    rm -rf pyram
    rm -rf Acoustics-Toolbox
    sudo rm -rf /opt/build/at
    sudo sed -i '/\/opt\/build\/at\/bin/d' ~/.bashrc
    
You have to manually uninstall dependencies if you want to.

## Roadmap

| To do                                                |        Status         |                                      Comments | 
|:-----------------------------------------------------|:---------------------:|----------------------------------------------:|
| Make oalib installation works properly               | Done                  |                                               |
| Update deprecated pyram types                        | Done                  |                                               |
| Simplify acoustic env() in arlpy                     | Done                  |                                               |
| Make a simple installation process                   | Done                  |                                               |
| Compatibility with Spyder                            | Done                  |                                               |
| Add basic plots                                      | Done                  |                                               |
| Add Wenz curves simulator                            | Done                  | docs_uac/WenzCurves.pdf                       |
| Add common sound profile plot                        | Done                  |                                               |
| Add plot PSD func in dB re 1uPa/vHz for rec signals  | Done                  |                                               |
| Add statistical PSD in dB re 1uPa/vHz                | Done                  |                                               |
| Add spectro func in dB re 1uPa/vHz for rec signals   | Done                  |                                               |
| Add SEL measurement                                  | Done                  |                                               |
| Add FRF for stationnary and transient signals        | Done                  |                                               |
| Handle source range and left propagation ? or not ?  | In progress ($\beta$) | maybe a bad idea                              |
| Update bellhop                                       | In progress ($\beta$) | docs_uac/Bellhop_userGuide.pdf                |
| Add pyram to arlpy                                   | In progress ($\beta$) | docs_uac/RAM.pdf                              |
| Add kraken to arlpy                                  | In progress ($\beta$) | docs_uac/KrakenNormalModeProgram_2001.pdf     |
| BugFix: Review FRF in stft mode                      | In progress           | Mean over best coherence only ?               |
| BugFix: PSD2PDF division problem                     | In progress           | RuntimeWarning: invalid value encountered ... |
| Manage all options for Bellhop, Kraken and RAM       | In progress           |                                               |
| Maintain up to date unittest and assert in arlpy     | In progress           |                                               |
| Maintain up to date function and class comments      | In progress           |                                               |
| Use optimal pade term in RAM                         | Not started           | docs_uac/RAM_pade_opti.pdf                    |
| Adjust output grid to exact requested one in RAM     | Not started           |                                               |
| Totally remove pandas                                | Not started           |                                               |
| Add channel simulator filter using IR ?              | Not started           |                                               |
| Add Krakenc to arlpy                                 | Not started           |                                               |
| Add earthquakes and explosions to Wenz model         | Not started           |                                               |
| Consider Mac/Windows compatibility                   | Not started           |                                               |
| ...                                                  | ...                   |                                               |
| Add scooter to arlpy                                 | Not started           |                                               |
| Add sparc to arlpy                                   | Not started           | docs_uac/SPARC_time_marched_ffp.pdf           |
| Consider using C++ version of bellhop                | Not started           | https://github.com/A-New-BellHope/bellhopcuda |

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
