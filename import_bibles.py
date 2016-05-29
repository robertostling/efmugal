# Script for importing texts from the bible corpus.
# Configuration is currently hard-coded at the bottom of this file.

import os.path
import glob
from collections import Counter


class BibleText:
    def __init__(self, name):
        self.name = name

    def from_file(self, filename, transform):
        verses = {}
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'): continue
                fields = line.strip().split('\t')
                if len(fields) != 2: continue
                verses[fields[0]] = list(map(transform, fields[1].split()))
        self.verses = verses


class ParallelText:
    def __init__(self, name, sents):
        self.name = name
        self.vocab = sorted(
                {token for sent in sents
                       if not sent is None
                       for token in sent})
        self.index = {token:i for i,token in enumerate(self.vocab)}
        self.sents = [
                None if sent is None else [self.index[token] for token in sent]
                for sent in sents]

    def write(self, f, text_id):
        f.write('%d %d %d\n' % (text_id, len(self.sents), len(self.vocab)))
        for sent in self.sents:
            if sent is None:
                f.write('0\n')
            else:
                f.write(' '.join(map(str, [len(sent)] + sent)) + '\n')


class BibleCorpus:
    def __init__(self, path):
        self.path = path
        self.texts = []
        self.parallel_texts = []

    def read_texts(self, pred=lambda name: True):
        def get_name(filename):
            return os.path.splitext(os.path.basename(filename))[0]

        pattern = os.path.join(self.path, '*-x-bible*.txt')
        filenames = sorted([filename for filename in glob.glob(pattern)
                            if pred(get_name(filename))],
                           key=get_name)

        def read_text(filename):
            text = BibleText(get_name(filename))
            text.from_file(filename, str.lower)
            return text

        self.texts = [read_text(filename) for filename in filenames]

    def sentence_align(self, min_ratio=0.75):
        min_count = int(min_ratio * len(self.texts))
        all_verses = [nr for text in self.texts
                         for nr in text.verses.keys()]
        verse_count = Counter(all_verses)
        verses = sorted(nr for nr, n in verse_count.items() if n >= min_count)
        self.verses = verses
        for i,text in list(enumerate(self.texts)):
            t = ParallelText(text.name, [text.verses.get(nr) for nr in verses])
            self.texts[i] = None
            self.parallel_texts.append(t)
        self.texts = None

    def write(self, path):
        try:
            os.mkdir(path)
        except FileExistsError:
            print('Warning: directory %s exists' % path)

        with open(os.path.join(path, 'verses.txt'), 'w') as f:
            f.write('\n'.join(self.verses) + '\n')

        with open(os.path.join(path, 'index.txt'), 'w') as index_f:
            index_f.write('%d\n' % len(self.parallel_texts))
            for text_id, text in enumerate(self.parallel_texts):
                filename = os.path.join(
                        path, '%d-%s.txt' % (text_id, text.name))
                index_f.write(os.path.abspath(filename) + '\n')
                with open(filename, 'w', encoding='ascii') as f:
                    text.write(f, text_id)

                filename = os.path.join(
                        path, '%d-%s.voc' % (text_id, text.name))
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(text.vocab) + '\n')


if __name__ == '__main__':
    import sys
    corpus = BibleCorpus(sys.argv[1])
    #corpus.read_texts(lambda name: name.startswith('swe'))
    corpus.read_texts()
    corpus.sentence_align()
    corpus.write('bibles-all')

