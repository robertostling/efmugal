import sys
import numpy as np

# Script to create a text similar to a parallel text (as created by
# import_bibles.py) but with tokens chosen from a uniform random distribution.
#
# Example:
#   random_like.py 5000 bibles/*bible*.txt >bibles/random.txt

def mean_lens(filenames):
    def file_lens(filename):
        with open(filename, 'r') as f:
            next(f)
            lens = [int(line.split()[0]) for line in f]
        return lens
    return np.array(list(map(file_lens, filenames))).mean(0).astype(np.int32)


if __name__ == '__main__':
    voc_size = int(sys.argv[1])
    lens = mean_lens(sys.argv[2:])
    n_sents = len(lens)
    print('0 %d %d' % (n_sents, voc_size))
    for n in lens:
        print(' '.join(map(str,
            [n] + list(np.random.randint(0, voc_size, (n,))))))

