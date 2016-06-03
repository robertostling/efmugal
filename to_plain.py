import sys
import os.path

# Convert an encoded file to plain text.
#
# Given test.txt or test.txt.out, and test.voc, use this like:
#   to_plain.py test.txt.out

def read_numeric(f):
    next(f)
    for line in f:
        yield list(map(int, line.split()[1:]))

def convert(filename):
    base = filename
    while True:
        base, ext = os.path.splitext(base)
        assert ext
        if ext == '.txt': break
    voc_filename = base + '.voc'
    with open(voc_filename, 'r', encoding='utf-8') as f:
        voc = f.read().split()
    with open(filename, 'r') as f:
        for ns in read_numeric(f):
            print(' '.join(voc[n] for n in ns))

if __name__ == '__main__':
    convert(sys.argv[1])

