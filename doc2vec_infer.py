#!/usr/bin/env python3
import argparse

from gensim.models import Doc2Vec


def main(argv=None):
    args = parse_args(argv)

    import utils

    validator = utils.extension_validator(
        args.wanted_extensions, args.unwanted_extensions)

    all_filenames = utils.all_files_iter(args.input_directory)
    wanted_filenames = utils.filter_filenames(all_filenames, validator)

    corpus = utils.TokenizedCorpusIter(wanted_filenames, args.encoding)
    model = Doc2Vec.load(args.model)
    docvecs_dict = model.infer_vectors(corpus, model, args.steps)
    utils.save_docvecs_as_csv(args.output, docvecs_dict, model.vector_size)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    def add(where, option, *args, **kwargs):
        where.add_argument(option, *args, **kwargs, help=HELP.get(option))

    add(parser, '--steps', type=int, required=True)
    add(parser, '--model', required=True)
    add(parser, '--input-directory', required=True)
    add(parser, '--output', metavar='FILENAME', required=True)
    add(parser, '--encoding', default='utf-8')

    input_group = parser.add_mutually_exclusive_group()
    add(input_group, '--wanted-extensions', metavar='EXT', nargs='+')
    add(input_group, '--unwanted-extensions', metavar='EXT', nargs='+')

    return parser.parse_args(argv)


DESCRIPTION = 'Infer embeddings for a set of tokenized documents using a trained doc2vec model.'

HELP = dict()
HELP['--steps'] = 'number of iterations for each document'
HELP['--model'] = 'filename of trained doc2vec model'
HELP['--input-directory'] = 'directory containing the tokenized documents'
HELP['--output'] = 'filename for result embeddings, in CSV format'
HELP['--encoding'] = 'document file encoding. Default is utf-8, but some datasets need latin-1'

HELP['--wanted-extensions'] = 'if specified, only process files with these extensions'
HELP['--unwanted-extensions'] = 'if specified, ignore files with these extensions'


if __name__ == '__main__':
    main()
