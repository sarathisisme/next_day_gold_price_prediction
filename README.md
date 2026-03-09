[![Shipping files](https://github.com/neuefische/ds-time-series/actions/workflows/workflow-05.yml/badge.svg?branch=main&event=workflow_dispatch)](https://github.com/neuefische/ds-time-series/actions/workflows/workflow-05.yml)

# Time Series

In this repo we will have a look at time series.

## Task

Please work in pairs through all the notebooks in this particular order:

1. [Intro to EDA with Time Series](01_Intro_EDA_Time_Series.ipynb)
1. [Time Series Stock Price Example](02_Time_Series_Stock_Price.ipynb)
1. [Time Series Moving average animation](03_Time_Series_Moving_Average.ipynb)
1. [Time Series Simulation](04_Time_Series_Decompose_Smoothing.ipynb)


## Set up your Environment

Please make sure you have forked the repo and set up a new virtual environment. For this purpose you can use the following commands:

The added [requirements file](requirements.txt) contains all libraries and dependencies we need to execute the time series notebooks.

*Note: If there are errors during environment setup, try removing the versions from the failing packages in the requirements file. M1 shizzle.*

make sure to install **ffmpeg** if you haven't done it before.

 - Check the **ffmpeg version**  by run the following commands:
    ```sh
    ffmpeg -version
    ```
    If you haven't installed it yet, begin at `step_1`. Otherwise, proceed to `step_2`.


### **`macOS`** type the following commands : 

- We have also added a [Makefile](Makefile) which has the recipe called 'setup' which will run all the commands for setting up the environment.
Feel free to check and use if you are tired of copy pasting so many commands.

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```
Or ....
- `Step_1:` Update Homebrew and install ffmpeg by following commands:
    ```sh
    brew update
    brew install ffmpeg
    ```
  Restart Your Terminal and than check the **ffmpeg version**  by run the following commands:
     ```sh
    ffmpeg -version
    ```
- `Step_2:` Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
### **`WindowsOS`** type the following commands :

- `Step_1:` Update Chocolatey and install ffmpeg by following commands:
    ```sh
    choco upgrade chocolatey
    choco install ffmpeg
    ```
    Restart Your Terminal and then check the **ffmpeg version**  by running the following commands:
     ```sh
    ffmpeg -version
    ```

- `Step_2:` Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

  
  
