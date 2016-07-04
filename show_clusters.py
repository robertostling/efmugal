import os.path
from collections import defaultdict

def read_links(filename):
    with open(filename, 'r') as f:
        return [list(map(int, line.split())) for line in f]

def read_voc(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def read_text(filename):
    with open(filename, 'r') as f:
        next(f)
        return [list(map(int, line.split()))[1:] for line in f]

def make_multilingual(concepts_filename, index_filename):
    with open(index_filename, 'r') as f:
        next(f)
        text_filenames = [line.strip() for line in f]
    voc_filenames = [os.path.splitext(s)[0]+'.voc' for s in text_filenames]
    links_filenames = [s+'.links' for s in text_filenames]
    concepts = read_text(concepts_filename)
    vocs = list(map(read_voc, voc_filenames))
    links = list(map(read_links, links_filenames))
    texts = list(map(read_text, text_filenames))
    assert len(vocs) == len(links)
    assert len(links) == len(texts)
    for i in range(len(concepts)):
        source_sent = concepts[i]
        translations = [[] for _ in source_sent]
        for text_voc, text_links, text in zip(vocs, links, texts):
            target_sent = [text_voc[x] for x in text[i]]
            for token, a in zip(target_sent, text_links[i]):
                translations[a].append(token)
        print(('\n'+'-'*72+'\n').join(
            ['\n'.join(words) for words in translations]))

if __name__ == '__main__':
    import sys
    make_multilingual(sys.argv[1], sys.argv[2])


