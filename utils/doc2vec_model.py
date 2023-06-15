import os

from gensim.models import Doc2Vec


class Doc2VecModel:
    def __init__(self, corpus, model_args, intersect_filename=None):
        self.corpus = corpus
        self.model_args = model_args
        self.intersect_filename = intersect_filename

        self.model = Doc2Vec(
            dbow_words=(intersect_filename is not None),
            **model_args
        )

        self.model.build_vocab(corpus)

        if intersect_filename is not None:
            self.model.intersect_word2vec_format(
                fname=intersect_filename,
                lockf=0,
                binary=intersect_filename.endswith('.bin')
            )

    def train(self):
        self.model.train(
            self.corpus,
            total_examples=self.model.corpus_count,
            epochs=self.model.iter
        )

    def save(self, where, csv=True):
        name = os.path.basename(where)
        os.makedirs(where, exist_ok=True)

        model_filename = os.path.join(where, name) + '.model'
        self.model.save(model_filename)

        if csv:
            csv_filename = os.path.join(where, name) + '.docvecs.csv'
            docvecs = docvecs_as_dict(self.model)
            save_docvecs_as_csv(csv_filename, docvecs, self.model_args['size'])

        settings_filename = os.path.join(where, 'settings.txt')
        with open(settings_filename, 'w', encoding='utf-8') as f:
            for key, value in self.model_args.items():
                print('{}: {}'.format(key, value), file=f)

            print('intersect: {}'.format(self.intersect_filename), file=f)


def save_docvecs_as_csv(filename, docvecs, vector_size, csv_header=None):
    """Takes a dict containing doctag: docvec data and saves as a CSV file.
    The default CSV header is 'd1,d2,d3,(...),dVSIZE,class_attr'.

    Arguments:
    filename -- output filename
    docvecs -- dict containing {tag: vector} pairs
    csv_header -- optionally specify CSV header to be used."""
    HEADER = ['d' + str(i + 1) for i in range(vector_size)] + ['class_attr']

    if csv_header is None:
        csv_header = HEADER

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        print(*csv_header, sep=',', file=f)

        for doctag, docvec in docvecs.items():
            print(*docvec, doctag, sep=',', file=f)


def docvecs_as_dict(model):
    return {t: model.docvecs[t] for t in model.docvecs.doctags}


def infer_vectors(corpus, model, steps):
    return {
        doc.tags[0]: model.infer_vector(doc.words, steps=steps)
        for doc in corpus
    }
