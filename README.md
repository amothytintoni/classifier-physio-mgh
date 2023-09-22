# mgh-data-labelling
hi, 
This classifier helps in distinguishing good and bad physiological data extracted from Respiree sensor.

btw so far we can only take in .csv files :) <br>

### How to run/use classifier: <br>
##### 1. Downloading:
&nbsp;&nbsp;&nbsp;&nbsp;Download your model of choice (e.g. xgb2.model), `test_main.py`, `test_funcs.py`, 
`functions_for_classifier.py`, `classifier_environment.yml` (and `sample_data.csv` if you need a sample) and put them 
in one directory. <br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;Required Modules: `numpy, pandas, xgboost, matplotlib, datetime`<br>

##### 2. Inputting: 
    
&nbsp;&nbsp;&nbsp;&nbsp;CLI inputting
    
    format: test_main.py [-h] [-f INPUT_FILE] [-outdir OUTPUT_DIR] [-p INPUT_PATH] [-t OUTPUT_THRESHOLD] [-o] 
    [-s] [-m MODEL_PATH] [-plot] [-v VERBOSITY] [-minf MIN_FRAMES_DAILY] [-l LIMIT_FLOOR_THR] [-ss STEP_SIZE] 
    [-trans] [-proba] [-outname OUTPUT_FILENAME]
    
    sample: python test_main.py -f sample_data.csv -t 0.5 -o -s -m xgb3.model
    
    explanation:
        a. If your input is a single file and it is in the same directory as the classifier, then only 
        enter -f your_file_name.
        b. If your input is a single file and it is not in the same directory as the classifier, then 
        enter -f your_file_name and -p your_directory.
        c. If your input is multiple file enclosed in a folder, then only enter -p your_directory.
        
    For a more detailed sample run, see the end of this readme.
    Note that the arguments in [] are optional, except for -f INPUT_FILE or -p INPUT_PATH: you have to 
    enter at least one.


##### 3. Outputting: 
There are a few output options:<br>
&nbsp;&nbsp;&nbsp;&nbsp;a. Patient-wise split for output files: `-s split_patient_output`<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Classifier defaults to outputting one big file. If you 
want to print one output file per patient, include -s in the argument.<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;b. Override / Overwrite: `-o, --override`<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -o Overrides existing PD Validity Labels. Throws an 
error if there is no existing PD Validity column.<br><br>



### If you want to train your own model: <br>
##### 1. Downloading: <br>
&nbsp;&nbsp;&nbsp;&nbsp;Download `train_main.py`, `train_funcs.py`, `functions_for_classifier.py`, 
`classifier_environment.yml` (and `sample_trainset.csv` if you need a sample train data) and put 
them in one directory. <br>

##### 2. Inputting train data:
&nbsp;&nbsp;&nbsp;&nbsp;The train and test data you provide have to contain/satisfy:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - all 7 features used in classifier (feature list below)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - existing 'PD Validity' column<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - userId, dateTime, dashboardMode column<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - train and test must be mutually exclusive<br>


CLI inputting <br>

```
format: train_main.py [-h] [-f INPUT_TRAIN] [-p INPUT_TRAIN_PATH] [-ff INPUT_TEST] [-pp INPUT_TEST_PATH]
[-t OUTPUT_THRESHOLD] [-o] [-s] [-md MAX_DEPTH] [-rv REG_VALUE] [-rm REG_MODE] [-eta ETA] [-g GAMMA] 
[-ev EVAL_METRIC] [-n NUM_ROUND] [-sw SCALE_POS_WEIGHT] [-diff] [-score] [-save SAVE_MODEL_NAME] [-r] 
[-plot] [-v VERBOSITY] [-trans TRANSLATE]

sample: python train_main.py -f sample_trainset.csv -p fnfpcheck -ff sample_unseen.csv -pp fnfpcheck 
-s -md 8 -rv 40 -rm l1 -ev auc -n 100 -diff -score -save traintry.model

explanation:
    a. If your input is a single file and it is in the same directory as the classifier, 
       then only enter -f your_train_file_name for train, and -ff your_test_file_name for test.
    b. If your input is a single file and it is not in the same directory as the classifier, 
       then enter -f your_train_file_name and -p your_train_directory. (-ff and -pp for test)
    c. If your input is multiple file enclosed in a folder, then only enter -p your_train_directory, or 
       -pp your_test_directory.
        
For a more detailed sample run and explanation on -minf (min_frames_daily) mode and its parameters, see the end of this readme.
Note that the arguments in [] are optional, except for -f INPUT_FILE or -p INPUT_PATH: you have to 
enter at least one.
```
##### 3. The outputs:
    a. Model
    I recommend that you name your model with a .model extension for uniformity and identification purposes. 
    Your trained model will be outputted as the name you input for -save argument
    
    b. Diff file
    This file is a compilation of where classifier and existing labels disagree. Useful for confidence checking.
    
    c. Updated train and test file
    As with test output, you can choose to output one big file which contains predicted train and test data, 
    or split it patient-wise using the -s argument

<br>
Do let me know if i can be of further assistance at <u>timothy.antoni@gmail.com</u><br>


#### Full sample test run:
format: `test_main.py [-h] [-f INPUT_FILE] [-outdir OUTPUT_DIR] [-p INPUT_PATH] [--output_threshold OUTPUT_THRESHOLD]
[--override] [--split_patient_output] --model_path MODEL_PATH [--importance_plot] [-v VERBOSITY]
[-minf MIN_FRAMES_DAILY] [-l LIMIT_FLOOR_THR] [-ss STEP_SIZE] [-trans] [-proba] [-outname OUTPUT_FILENAME]`<br>
```
cwd> python test_main.py -f sample_data.csv -t 0.5 -o -s -m 3 -v 2

initializing completed
model used = xgb3.model
output_threshold = 0.5
data length before na filter = 175
clean test data length = 165
preprocessing done
prediction in progress...
done predicting, writing now...
file outputted as 152_2022-10-12_predicted_with_override.csv
file outputted as 247_2022-10-12_predicted_with_override.csv
file outputted as 248_2022-10-12_predicted_with_override.csv
file outputted as 155_2022-10-12_predicted_with_override.csv
file outputted as 156_2022-10-12_predicted_with_override.csv
file outputted as 249_2022-10-12_predicted_with_override.csv
file outputted as 154_2022-10-12_predicted_with_override.csv
thanks for using respiretclassifier dawg
```

##### Full help for testing
format: `test_main.py [-h] [-f INPUT_FILE] [-outdir OUTPUT_DIR] [-p INPUT_PATH] [--output_threshold OUTPUT_THRESHOLD]
[--override] [--split_patient_output] --model_path MODEL_PATH [--importance_plot] [-v VERBOSITY]
[-minf MIN_FRAMES_DAILY] [-l LIMIT_FLOOR_THR] [-ss STEP_SIZE] [-trans] [-proba] [-outname OUTPUT_FILENAME]` <br><br>
  ```
  -h, --help            show this help message and exit
  -f INPUT_FILE         Type of input. Enter "-f your_file_name" for a single input file, or "-p your_directory" for a
                        directory target
  -p INPUT_PATH         Type of input. Enter -f your_file_name for a single input file, or -p your_directory for a
                        directory target
  -t OUTPUT_THRESHOLD, --output_threshold OUTPUT_THRESHOLD
                        If you want to change the output_threshold. Input range: (0,1). Higher threshold means more
                        confidence on pd=1 predictions, and vice versa. Default = 0.6
  -o, --override        --override will override existing PD Validity. Only works if there is an existing "PD
                        Validity" column. Default = False
  -s, --split_patient_output
                        --split_patient_output will split the output for each patients in the test file. Do not enter
                        if you want the output to be a single big file. Default = False
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path to the model that you would like to use. Default = xgb_4.0.model
  -plot, --importance_plot
                        Write Feature Importance Plots to files. Default = False
  -v VERBOSITY, --verbose VERBOSITY
                        Level of verbosity. 0 = no text at all, 1 = prints model and data details, 2 = print all
                        progress
  -minf MIN_FRAMES_DAILY, --min_frames_daily MIN_FRAMES_DAILY
                        Classifier will attempt to get the minimum number of valid daily frames specified, sacrificing specificity as a consequence. Default = None
  -l LIMIT_FLOOR_THR, -limit_floor_thr LIMIT_FLOOR_THR
                        To obtain a minimum of 20-frames per day, how low will you allow the final layer threshold to be? Rough guide: At 0.6 threshold, Specificity is 95.5% and Sensitivity is 81%, at 0.5 threshold spec is 94% and sens is 84%, at 0.4 spec is 90.5% and sens is 87%, at 0.3 spec is 85% and sens is 90%. Only applicable in -minf mode. Default = 0.45
  -ss STEP_SIZE, --step_size STEP_SIZE
                        Output threshold's decrement step size. Only applicable in -minf mode. Default = 0.05
  -trans, --translate
                        Translate feature names to everyday language. Useful for explainer plot. Default = False
  -proba                Prints probability score (pred_proba). Default = False
  -outname, --output_filename
                        The desired output file name.
   ``` 


##### Full help for training <br>
format: `train_main.py [-h] [-f INPUT_TRAIN] [-p INPUT_TRAIN_PATH] [-ff INPUT_TEST] [-pp INPUT_TEST_PATH]
[-t OUTPUT_THRESHOLD] [-o] [-s] [-md MAX_DEPTH] [-rv REG_VALUE] [-rm REG_MODE] [-eta ETA] [-g GAMMA] 
[-ev EVAL_METRIC] [-n NUM_ROUND] [-sw SCALE_POS_WEIGHT] [-mcw MIN_CHILD_WEIGHT][-diff] [-score] 
[-save SAVE_MODEL_NAME] [-r] [-plot] [-v VERBOSITY] [-trans]` <br>
  
  ```
  -h, --help            show this help message and exit
  -f INPUT_TRAIN        Type of input. Enter "-f your_trainfile_name" for a single input file, or "-p your_directory"
                        for a directory target
  -p INPUT_TRAIN_PATH   Type of input. Enter -f your_trainfile_name for a single input file, or -p your_directory for
                        a directory target
  -ff INPUT_TEST        Type of input. Enter "-f your_testfile_name" for a single input file, or "-p your_directory"
                        for a directory target
  -pp INPUT_TEST_PATH   Type of input. Enter -f your_testfile_name for a single input file, or -p your_directory for a
                        directory target
  -t OUTPUT_THRESHOLD, --output_threshold OUTPUT_THRESHOLD
                        If you want to change the output_threshold. Input range: (0,1). Higher threshold means more
                        confidence on pd=1 predictions, and vice versa. Default = 0.6
  -o, --override        --override will override existing PD Validity. Only works if there is an existing "PD
                        Validity" column. Default = False
  -s, --split_patient_output
                        --split_patient_output will split the output for each patients in the test file. Do not enter
                        if you want the output to be a single big file. Default = False
  -md MAX_DEPTH, --max_depth MAX_DEPTH
                        parameter for the classifier. Default = 7
  -rv REG_VALUE, --reg_value REG_VALUE
                        parameter for the classifier. Default = 30
  -rm REG_MODE, --reg_mode REG_MODE
                        parameter for the classifier. Default = l1
  -eta ETA              parameter for the classifier. Default = 0.1
  -g GAMMA, --gamma GAMMA
                        parameter for the classifier. Default = 1.0
  -ev EVAL_METRIC, --eval_metric EVAL_METRIC
                        parameter for the classifier. Default = auc
  -n NUM_ROUND, --num_round NUM_ROUND
                        parameter for the classifier. Default = 150
  -sw SCALE_POS_WEIGHT  parameter for the classifier. Default = 1
  -mcw MIN_CHILD_WEIGHT
                        min_child_weight parameter for the classifier. Default = 1
  -diff                 write disagreements between existing label and classifier's prediction to a csv file. Default
                        = False
  -score                write model's scores to a csv file. Default = False
  -save SAVE_MODEL_NAME
                        write model's scores to a file. Default = newest.model
  -r                    write predicted train and test data into a csv file. Default = False
  -plot, --importance_plot
                        Write Feature Importance Plots to files. Default = False
  -v VERBOSITY, --verbose VERBOSITY
                        Level of verbosity. 0 = no text at all, 1 = prints model and data details, 2 = print all
                        progress
  -trans, --translate   Translate feature names to everyday language. Useful for explainer plot. Default = False
  ```
  
line to check flatline index:
`print(flatline_checker(user_id='154', startDateTime='2022-06-10 04:35:41')) should give out 0.235 for flatline_tol_diff = 0, or 0.281 for flatline_tol_diff = 1`
