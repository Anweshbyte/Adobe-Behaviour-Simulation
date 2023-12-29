# Behaviour Simulation
&emsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GXOhJ7rDmb-6ijpZzxTroug7I_ACbiNE?usp=sharing) 
---
## Instructions
Following are the setup and running instructions for this script. 
If the script FAILS to run, for any reason, please refer to [this](https://colab.research.google.com/drive/1GXOhJ7rDmb-6ijpZzxTroug7I_ACbiNE?usp=sharing) colab instance for running the model.
### Setup
Switch to the `Behaviour_Simulation/` directory and install the dependencies :-
```
pip install -r requirements.txt
```
Next install the model
```
gdown 1EhzCuqE37eepqpM2vSfq7c6PEWjXXJxf
```
Finally your `/Behaviour_Simulation/` directory should look as follows
 ```
Behaviour_Simulation
    ├── batchGenerator.py
    ├── cleaner.py
    ├── Clip_model_All_my_sample.h5
    ├── main.py
    ├── media_proc.py
    ├── norms.json
    ├── READMe.md
    ├── requirements.txt
    └── utils.py
 ```

### Running the model
- Make sure you are in the `/Behaviour_Simulation/` directory and download the dataset here.
- Run the command below for getting the like predictions :-
```
python main.py "<$path_to_excel_file>" 
```
- The results can be found as `results.xlsx` in the `/outputs/` directory.


<!-- - Run ```pip install -r requirements.txt``` to install the dependencies
- Download the dataset in the directory
- Run ```python main.py path/to/your/dataset.xlsx``` to save the results as ```Submission.csv``` -->

