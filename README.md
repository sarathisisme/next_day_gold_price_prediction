# Next day gold price prediction

The app model is trained every day (with new data) in order to make the next day gold price prediction

## App
https://sara-gold-forecast.streamlit.app/

## Jupyter Notebooks
- After checking out the repository please follow the steps in below section to set up the environment
- To open Jupyter Lab enter this after environment activation
    ```sh
    jupyter-lab
    ```
- You can use all the jupyter notebooks. Except news_sentiment.ipynb because it does not have the data files as they are too big for GitHub
- Baseline_Trend.ipynb can be run to get results for baseline model
- ARMA.ipynb can be run to get results of improved model
- XGBoost_With_All_Exo_Wo_Sentiment.ipynb contains advanced model without news sentiment
- XGBoost_With_All_Exo.ipynb contains advanced model wtih news sentiment
- daily_run.py is the script that runs every day and train model with new data
- gold_predictions.csv is where the results of the above script are stored
- app.py is for the web app
- nlp_data_mining.ipynb is used to get news text from new york times API

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

  
  
