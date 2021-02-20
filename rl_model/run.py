from web_dqn import start_learning
import sys
import argparse


def createParser ():
    """
    ArgumentParser
    """
    parser = argparse.ArgumentParser()
    #parser.add_argument('-mn', '--min_num', default=5, type=int,\
    #    help='min number of common results')
    #parser.add_argument('-t', '--tests', default=None, type=str,\
    #    help='list of tests as -t 1,2,11')
    parser.add_argument('-c', '--continue_training', dest='continue_training', action='store_true')
    return parser


if __name__ == "__main__":

    parser = createParser()
    arguments = parser.parse_args(sys.argv[1:])
    print("continue_training:", arguments.continue_training)
    start_learning(continue_training=arguments.continue_training)