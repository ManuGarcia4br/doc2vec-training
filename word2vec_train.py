#!/usr/bin/env python3
import os
import argparse
from multiprocessing import cpu_count


def main(argv=None):
    args = parse_args(argv)

    import utils

    if not args.exist_ok:
        check_exists(args.output_directory)

    model_args = {
        'dm': args.dm,
        'workers': args.workers,
        'iter': args.epochs,
        'window': args.window,
        'min_count': args.min_count,
        'size': args.vector_size,
    }

    validator = utils.extension_validator(
        args.wanted_extensions, args.unwanted_extensions)

    all_filenames = utils.all_files_iter(args.input_directory)
    wanted_filenames = utils.filter_filenames(all_filenames, validator)

    corpus = utils.TokenizedCorpusIter(wanted_filenames, args.encoding)
    model = utils.Doc2VecModel(corpus, model_args, args.intersect)
    model.train()
    model.save(args.output_directory, csv=not args.no_csv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    def add(where, option, *args, **kwargs):
        where.add_argument(option, *args, **kwargs, help=HELP.get(option))

    add(parser, '--dm', default=0, const=1, action='store_const')
    add(parser, '--workers', type=int, default=cpu_count())
    add(parser, '--epochs', type=int, required=True)
    add(parser, '--window', type=int, required=True)
    add(parser, '--min-count', type=int, required=True)
    add(parser, '--vector-size', type=int, required=True)
    add(parser, '--intersect', metavar='WORD2VEC_MODEL')

    add(parser, '--exist-ok', action='store_true')
    add(parser, '--encoding', default='utf-8')
    add(parser, '--no-csv', action='store_true')

    add(parser, '--input-directory', metavar='DIRECTORY', required=True)
    add(parser, '--output-directory', metavar='DIRECTORY', required=True)

    input_group = parser.add_mutually_exclusive_group()
    add(input_group, '--wanted-extensions', metavar='EXT', nargs='+')
    add(input_group, '--unwanted-extensions', metavar='EXT', nargs='+')

    return parser.parse_args(argv)


def check_exists(directory):
    try:
        num_files = len(os.listdir(directory))
        if num_files != 0:
            raise FileExistsError(
                "Directory '{}' exists and is not empty".format(directory))
        # Directory empty -> ok
    except FileNotFoundError:
        pass


DESCRIPTION = 'Train a doc2vec model from a tokenized corpus.'

HELP = dict()
HELP['--dm'] = 'use DM training algorithm instead of DBOW'
HELP['--workers'] = 'use these many worker threads to train the model. Default: cpu count'
HELP['--epochs'] = 'number of iterations through the corpus'
HELP['--window'] = 'the maximum distance between the current and predicted word within a sentence'
HELP['--min-count'] = 'ignores all words with total frequency lower than this'
HELP['--vector-size'] = 'dimensionality of the feature vectors'
HELP['--intersect'] = 'initialize word weights using this word2vec model. Binary models must have .bin extension'

HELP['--exist-ok'] = 'do not throw an error if output-directory is already populated'
HELP['--encoding'] = 'document file encoding. Default is utf-8, but some datasets need latin-1'
HELP['--no-csv'] = 'do not create csv with embeddings'

HELP['--input-directory'] = 'directory containing the corpus'
HELP['--output-directory'] = 'directory to store output (model, embeddings, and parameters)'

HELP['--wanted-extensions'] = 'if specified, only process files with these extensions'
HELP['--unwanted-extensions'] = 'if specified, ignore files with these extensions'


if __name__ == '__main__':
    main()
