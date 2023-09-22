import train_funcs as cl_func
import argparse


def train(input_train,
          path_trainset,
          input_test,
          path_testset,
          output_threshold,
          split_patient_output,
          override,
          print_diff_to_file,
          print_score_to_file,
          print_predict_result,
          max_depth,
          reg_value,
          eta,
          gamma,
          reg_mode,
          eval_metric,
          scale_pos_weight,
          num_round,
          save_model_name,
          importance_plot,
          min_child_weight,
          verbosity,
          translate,
          output_dir='.',
          ):

    clf = cl_func.Classifier_xgb(
        input_train=input_train,
        path_trainset=path_trainset,
        input_test=input_test,
        path_testset=path_testset,
        output_threshold=output_threshold,
        split_patient_output=split_patient_output,
        override=override,
        print_diff_to_file=print_diff_to_file,
        print_score_to_file=print_score_to_file,
        print_predict_result=print_predict_result,
        max_depth=max_depth,
        reg_value=reg_value,
        eta=eta,
        gamma=gamma,
        reg_mode=reg_mode,
        eval_metric=eval_metric,
        scale_pos_weight=scale_pos_weight,
        num_round=num_round,
        save_model_name=save_model_name,
        importance_plot=importance_plot,
        min_child_weight=min_child_weight,
        verbosity=verbosity,
        translate=translate,
        output_dir=output_dir,
    )

    clf.process_train()
    clf.train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Classifier for Physiological Data Validity')
    parser.add_argument('-f', dest='input_train', required=False,
                        help='Type of input. Enter "-f your_trainfile_name" for a single input file, or "-p your_directory" for a directory target')
    parser.add_argument('-p', dest='input_train_path', required=False,
                        help='Type of input. Enter -f your_trainfile_name for a single input file, or -p your_directory for a directory target')

    parser.add_argument('-ff', dest='input_test', required=False,
                        help='Type of input. Enter "-f your_testfile_name" for a single input file, or "-p your_directory" for a directory target')
    parser.add_argument('-pp', dest='input_test_path', required=False,
                        help='Type of input. Enter -f your_testfile_name for a single input file, or -p your_directory for a directory target')

    parser.add_argument('-t', '--output_threshold', type=float,
                        default=0.6,
                        help='If you want to change the output_threshold. Input range: (0,1). Higher threshold means more confidence on pd=1 predictions, and vice versa. Default = 0.6')
    parser.add_argument('-o', '--override', dest='override', action='store_true',
                        help='--override will override existing PD Validity. Only works if there is an existing "PD Validity" column. Default = False')
    parser.add_argument('-s', '--split_patient_output', dest='split_patient_output', action='store_true',
                        help='--split_patient_output will split the output for each patients in the test file. Do not enter if you want the output to be a single big file. Default = False')
    parser.add_argument('-md', '--max_depth', dest='max_depth',
                        default=7,
                        help='parameter for the classifier. Default = 7')
    parser.add_argument('-rv', '--reg_value', dest='reg_value',
                        default=30,
                        help='parameter for the classifier. Default = 30')
    parser.add_argument('-rm', '--reg_mode', dest='reg_mode',
                        default='l1',
                        help='parameter for the classifier. Default = l1')
    parser.add_argument('-eta', dest='eta',
                        default=0.1,
                        help='parameter for the classifier. Default = 0.1')
    parser.add_argument('-g', '--gamma', dest='gamma',
                        default=1.0,
                        help='parameter for the classifier. Default = 1.0')
    parser.add_argument('-ev', '--eval_metric', dest='eval_metric',
                        default='auc',
                        help='parameter for the classifier. Default = auc')
    parser.add_argument('-n', '--num_round', dest='num_round',
                        default=150,
                        help='parameter for the classifier. Default = 150')
    parser.add_argument('-sw', dest='scale_pos_weight',
                        default=1,
                        help='parameter for the classifier. Default = 1')
    parser.add_argument('-mcw', dest='min_child_weight',
                        default=1,
                        help='min_child_weight parameter for the classifier. Default = 1')
    parser.add_argument('-diff', dest='print_diff_to_file', action='store_true',
                        help='write disagreements between existing label and classifier\'s prediction to a csv file. Default = False')
    parser.add_argument('-score', dest='print_score_to_file', action='store_true',
                        help='write model\'s scores to a csv file. Default = False')
    parser.add_argument('-save', dest='save_model_name', default='newest.model',
                        help='write model\'s scores to a file. Default = newest.model')
    parser.add_argument('-r', dest='print_predict_result', action='store_true',
                        help='write predicted train and test data into a csv file. Default = False')
    parser.add_argument('-plot', '--importance_plot', dest='importance_plot',
                        action='store_true', help='Write Feature Importance Plots to files. Default = False')
    parser.add_argument('-v', '--verbose', dest='verbosity', type=int,
                        default=1,
                        help='Level of verbosity. 0 = no text at all, 1 = prints model and data details, 2 = print all progress')
    parser.add_argument(
        '-trans',
        '--translate',
        dest='translate',
        action='store_true',
        help='Translate feature names to everyday language. Useful for explainer plot. Default = False')

    args = parser.parse_args()

    if ((args.input_test is not None) or (args.input_test_path is not None)):
        input_test = args.input_test
        path_testset = args.input_test_path
    else:
        raise Exception(
            'please enter the input mode: -ff [test file name] or -pp [path name]')

    if ((args.input_train is not None) or (args.input_train_path is not None)):
        input_train = args.input_train
        path_trainset = args.input_train_path
    else:
        raise Exception(
            'please enter the input mode: -f [train file name] or -p [path name]')

    train(input_train=input_train,
          path_trainset=path_trainset,
          input_test=input_test,
          path_testset=path_testset,
          override=args.override,
          split_patient_output=args.split_patient_output,
          output_threshold=args.output_threshold,
          print_diff_to_file=args.print_diff_to_file,
          print_score_to_file=args.print_score_to_file,
          print_predict_result=args.print_predict_result,
          max_depth=args.max_depth,
          reg_value=args.reg_value,
          eta=args.eta,
          gamma=args.gamma,
          reg_mode=args.reg_mode,
          eval_metric=args.eval_metric,
          scale_pos_weight=args.scale_pos_weight,
          num_round=args.num_round,
          save_model_name=args.save_model_name,
          importance_plot=args.importance_plot,
          min_child_weight=args.min_child_weight,
          verbosity=args.verbosity,
          translate=args.translate,
          )
