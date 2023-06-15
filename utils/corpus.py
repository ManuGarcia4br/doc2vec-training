from gensim.models.doc2vec import TaggedDocument


class TokenizedCorpusIter:
    def __init__(self, filenames, encoding):
        self.filenames = filenames
        self.encoding = encoding

    def __iter__(self):
        for filename in self.filenames:
            with open(filename, encoding=self.encoding) as f:
                lines = f.read().split('\n')
            sentences = [line.split(' ') for line in lines]
            words = [word for sent in sentences for word in sent]
            yield TaggedDocument(tags=[filename], words=words)
