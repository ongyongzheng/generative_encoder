"""
@author: Yong Zheng Ong
the main function for running the ge model
"""

import importlib
import argparse
import sys

if __name__ == "__main__":

    # argparser
    parser = argparse.ArgumentParser(description='Testing script', add_help=False)
    parser.add_argument('evaluation_name', type=str,
                        help='Name of the evaluation method to launch. To get \
                        the arguments specific to an evaluation method please \
                        use: eval.py evaluation_name -h')
    parser.add_argument('evaluation_id', type=str,
                        help='Identifier of the evaluation to launch. The result will be saved in "result/evaluation_id')

    # args for generative_encoder
    parser.add_argument('-position', type=int, required=False) # file name
    parser.add_argument('-with_ae', default=True, action='store_false')
    parser.add_argument('-gan_name', type=str, required=False)
    parser.add_argument('-ae_name', type=str, required=False)

    args = parser.parse_args()

    # validity check for args
    if args.evaluation_name in ['celeba']:
        assert args.position is not None, "position field should not be empty!"
        assert args.with_ae is not None, "with_ae field should not be empty!"

        module = importlib.import_module("tests.generative_encoder.generate_images")

    else:
        raise ValueError("evaluation name provided is invalid")

    print("Running " + args.evaluation_name)
    module.test(parser)