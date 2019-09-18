# efmugal
Efficient Multilingual Gibbs Aligner

This is an implementation of an improved version of
[Östling (2014)](https://www.aclweb.org/anthology/E/E14/E14-4024.pdf),
using a partially collapsed sampler for better parallelization.
The current code uses OpenMP for parallelizing sampling on a multi-core
system.

Preliminary experiments showed that mixing is too slow for this sampler to be
practical with more than a few dozen languages, so work has been abandoned.
The code is published here in case anyone wants to experiment with it.

Parts of the codebase has been turned into the
[efficient bitext aligner eflomal](https://github.com/robertostling/eflomal)
which can be used to perform pairwise word alignment -- in many cases a better
solution than using the Östling (2014) multilingual alignment method.

## Authors

Robert Östling and Claire De Maricourt.

## Data format

You need a multi-parallel corpus without gaps, i.e. translations of each of
`n_sentences` sentences to each of `n_languages` languages.

The text of each language is described by a file of this format:

    language_id n_sentences vocabulary_size
    sent1_length token1 token2 token3 [...]
    sent2_length token1 token2 token3 [...]
    [...] 

Where all of the items are decimal integers.

These should be summarized in a corpus description file of the following
format:

    n_languages
    language_1.file
    language_2.file
    [...]
    language_n.file 

Where `n_languages` is a decimal integer, and the filenames correspond to the
encoded texts for each language as described above.

Next, call the main program:

    ./efmugal corpus-description-file concept-initializations-file n_iterations 

The `concept-initializations-file` is of the same format as (and can indeed be
one of) the encoded text files. It could also be randomly intialized, but this
likely leads to suboptimal results. There is a script `random_like.py` for
producing random texts in the correct format.

