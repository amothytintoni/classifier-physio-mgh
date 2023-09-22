
import test_funcs as cl_func
import functions_for_classifier as funcs
import argparse


def test(input_test,
         path_testset,
         override,
         split_patient_output,
         output_threshold,
         model_path,
         importance_plot,
         verbosity,
         limit_floor_thr,
         step_size,
         translate,
         min_frames_daily,
         print_proba,
         output_filename,
         output_dir='.',
         ):
    clf = cl_func.Classifier_xgb(input_data_path=input_test,
                                 input_data_dir=path_testset,
                                 override=override,
                                 split_patient_output=split_patient_output,
                                 output_threshold=output_threshold,
                                 model_path=model_path,
                                 importance_plot=importance_plot,
                                 verbosity=verbosity,
                                 limit_floor_thr=limit_floor_thr,
                                 step_size=step_size,
                                 output_dir=output_dir,
                                 translate=translate,
                                 min_frames_daily=min_frames_daily,
                                 print_proba=print_proba,
                                 output_filename=output_filename,
                                 )

    clf.process_test()
    clf.predict_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classifier for Physiological Data Validity')
    parser.add_argument(
        '-f',
        dest='input_file',
        required=False,
        help='Type of input. Enter "-f your_file_name" for a single input file, or "-p your_directory" for a directory target'
    )
    parser.add_argument('-outdir', dest='output_dir', required=False,
                        default='.', help='Output directory path.')
    parser.add_argument(
        '-p',
        dest='input_path',
        required=False,
        help='Type of input. Enter -f your_file_name for a single input file, or -p your_directory for a directory target')
    parser.add_argument(
        '--output_threshold',
        '-t',
        type=float,
        required=False,
        default=0.6,
        help='If you want to change the output_threshold. Input range: (0,1). Higher threshold means more confidence on pd=1 predictions, and vice versa.'
    )
    parser.add_argument(
        '--override',
        '-o',
        action='store_true',
        help='--override will override existing PD Validity. Only works if there is an existing "PD Validity" column.')
    parser.add_argument(
        '--split_patient_output',
        '-s',
        action='store_true',
        help='--split_patient_output will split the output for each patients in the test file. Do not enter if you want the output to be a single big file'
    )
    parser.add_argument(
        '--model_path',
        '-m',
        type=str,
        required=True,
        help='Path to the model that you would like to use. Default = xgb_4.0.model')
    parser.add_argument(
        '--importance_plot',
        '-plot',
        action='store_true',
        help='Write Feature Importance Plots to files')
    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbosity',
        type=int,
        default=1,
        help='Level of verbosity. 0 = no text at all, 1 = prints model and data details, 2 = print all progress')
    parser.add_argument(
        '-minf',
        '--min_frames',
        dest='min_frames_daily',
        help='Classifier will attempt to get the minimum number of valid daily frames specified, sacrificing specificity as a consequence. Default = None')
    parser.add_argument(
        '-l',
        '--limit_floor_thr',
        dest='limit_floor_thr',
        default=0.45,
        type=float,
        help='To obtain a minimum of 20-frames per day, how low will you allow the final layer threshold to be? Rough guide: At 0.6 threshold, Specificity is 95.5% and Sensitivity is 81%, at 0.5 threshold spec is 94% and sens is 84%, at 0.4 spec is 90.5% and sens is 87%, at 0.3 spec is 85% and sens is 90%. Only applicable in when -minf is specified. Default = 0.45')
    parser.add_argument(
        '-ss',
        '--step_size',
        dest='step_size',
        type=float,
        default=0.05,
        help='Output threshold\'s decrement step size. Only applicable in when -minf is specified. Default = 0.05')
    parser.add_argument(
        '-trans',
        '--translate',
        dest='translate',
        action='store_true',
        help='Translate feature names to everyday language. Useful for explainer plot. Default = False')
    parser.add_argument(
        '-proba',
        dest='print_proba',
        action='store_true',
        help='Prints probability score (pred_proba). Default = False')
    parser.add_argument(
        '-outname',
        '--output_filename',
        dest='output_filename',
        default=None,
        help='The desired output file name.'
    )

    args = parser.parse_args()

    if ((args.input_file is not None) or (args.input_path is not None)):
        input_test = args.input_file
        path_testset = args.input_path
    else:
        raise Exception('please enter the input mode: -f [file name] or -p [path name]')

    test(input_test=input_test,
         path_testset=path_testset,
         override=args.override,
         split_patient_output=args.split_patient_output,
         output_threshold=args.output_threshold,
         model_path=args.model_path,
         importance_plot=args.importance_plot,
         verbosity=args.verbosity,
         limit_floor_thr=args.limit_floor_thr,
         step_size=args.step_size,
         translate=args.translate,
         min_frames_daily=args.min_frames_daily,
         print_proba=args.print_proba,
         output_filename=args.output_filename,
         output_dir=args.output_dir,
         )
