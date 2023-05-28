#!/usr/bin/env python3
import os
import sys
import argparse


word_tokenize = None
sent_tokenize = None


def main(argv=None):
    args = parse_args(argv)

    import utils
    import nltk.tokenize

    global word_tokenize
    global sent_tokenize

    word_tokenize = nltk.tokenize.word_tokenize
    sent_tokenize = nltk.tokenize.sent_tokenize

    validator = utils.extension_validator(
        args.wanted_extensions, args.unwanted_extensions)

    all_filenames = utils.all_files_iter(args.input_directory)
    wanted_filenames = utils.filter_filenames(all_filenames, validator)
    num_files = len(wanted_filenames)

    preprocess_args = [args.nltk_lang, args.keep_case, args.keep_diacritics]

    print('Processing', num_files, 'files', file=sys.stderr)
    errors = 0

    for input_filename in wanted_filenames:
        try:
            output_filename = os.path.join(
                args.output_directory,
                input_filename[len(args.input_directory) + 1:])

            with open(input_filename, encoding=args.encoding) as f:
                data = f.read()

            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

            with open(output_filename, 'w', encoding=args.encoding) as f:
                sentences = preprocess_document(data, *preprocess_args)

                for sentence in sentences:
                    print(*sentence, file=f)
        except Exception as e:
            print('[!]', input_filename, '-', e, file=sys.stderr)
            errors += 1

    print('Finished. {} files, {} errors.'.format(num_files, errors),
          file=sys.stderr)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    def add(where, option, *args, **kwargs):
        where.add_argument(option, *args, **kwargs, help=HELP[option])

    add(parser, '--nltk-lang', metavar='LANG', required=True)
    add(parser, '--keep-case', action='store_true')
    add(parser, '--keep-diacritics', action='store_true')
    add(parser, '--encoding', default='utf-8')

    add(parser, '--input-directory', metavar='DIR', required=True)
    add(parser, '--output-directory', metavar='DIR', required=True)

    input_group = parser.add_mutually_exclusive_group()
    add(input_group, '--wanted-extensions', metavar='EXT', nargs='+')
    add(input_group, '--unwanted-extensions', metavar='EXT', nargs='+')

    return parser.parse_args(argv)


def preprocess_document(text, nltk_lang, keep_case, keep_diacritics):
    '''Converts a document to a list of sentences, \
    where each sentence is a list of words.'''

    args = [nltk_lang, keep_case, keep_diacritics]

    sentences = sent_tokenize(text, nltk_lang)
    return [preprocess_sentence(s, *args) for s in sentences]


def preprocess_sentence(sentence, nltk_lang, keep_case, keep_diacritics):
    if not keep_case:
        sentence = sentence.lower()
    if not keep_diacritics:
        sentence = sentence.translate(DIACRITICS_TABLE)
    return word_tokenize(sentence, nltk_lang)


DIACRITICS_TABLE = str.maketrans(
    'áâãàéêíóõôúüçÁÂÃÀÉÊÍÓÕÔÚÜÇ',
    'aaaaeeiooouucAAAAEEIOOOUUC',
)


DESCRIPTION = 'Tokenize a corpus into sentences and words using NLTK.'

HELP = dict()
HELP['--nltk-lang'] = 'language used by word and sentence tokenizer'
HELP['--keep-case'] = "don't convert text to lowercase before tokenizing"
HELP['--keep-diacritics'] = "don't remove diacritics (e.g. ` ´ ~ ^) before tokenizing."
HELP['--encoding'] = 'document file encoding. Default is utf-8, but some datasets need latin-1'

HELP['--input-directory'] = 'directory containing the corpus'
HELP['--output-directory'] = 'directory to store tokenized corpus'

HELP['--wanted-extensions'] = 'if specified, only process files with these extensions'
HELP['--unwanted-extensions'] = 'if specified, ignore files with these extensions'


if __name__ == '__main__':
    main()
