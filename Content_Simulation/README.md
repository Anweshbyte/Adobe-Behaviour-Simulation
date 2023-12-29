# Content Simulation 
&emsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Zulk3BocFkqu1xTUbQwcvH7NqhYplHeZ?usp=sharing) 

---
## Instructions
Following are the setup and running instructions for this script.
If the script FAILS to run, for any reason, please refer to [this](https://colab.research.google.com/drive/1Zulk3BocFkqu1xTUbQwcvH7NqhYplHeZ?usp=sharing) colab instance for running the model

### Setup 
Switch to the `Content_Simulation/` directory and install the dependencies :-
```
pip install -r requirments.txt
``` 

### Running the model
- Make sure you are in the `Content_Simulation/` directory and download the dataset there.
- Run the command below for getting the like predictions :-
```
python main.py "<$path_to_excel_file>" 
``` 
- The results can be found as `results.xlsx` in the `outputs/` directory.

