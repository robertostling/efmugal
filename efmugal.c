#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include <time.h>

#define PRIlink     PRIu16
#define PRItoken    PRIu32
#define SCNlink     SCNu16
#define SCNtoken    SCNu32

typedef uint16_t link;
typedef uint32_t token;

#ifdef SINGLE_PRECISION
typedef float count;
#else
typedef double count;
#endif


#include "random.c"
#include "hash.c"

#define MAX_FIXED   7
#define MIN_DYNAMIC 4

#define MAKE_NAME(NAME) map_token_u32 ## NAME
#define INDEX_TYPE      size_t
#define KEY_TYPE        token
#define VALUE_TYPE      uint32_t
#define EMPTY_KEY       ((token)0xffffffffUL)
#define HASH_KEY        hash_u32_u32
#include "natmap.c"

struct sentence {
    link length;
    token tokens[];
};

struct text {
    int language;
    char *filename;
    size_t n_sentences;
    token vocabulary_size;
    struct sentence **sentences;
};

struct text_alignment {
    const struct text *source;
    const struct text *target;
    link **sentence_links;
    link *buf;
    struct map_token_u32 *source_count;
    count *inv_source_count_sum;
    count alpha;
};

struct corpus {
    size_t n_texts;
    struct text *source;
    struct text **texts;
    struct text_alignment **text_alignments;
};

double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return 1e-9*(double)ts.tv_nsec + (double)ts.tv_sec;
}

#define MIN(x,y)    (((x)<(y))?(x):(y))

static inline void transpose(
        count **dest,
        count *const*src,
        size_t src_rows,
        size_t src_cols,
        size_t step)
{
#pragma omp parallel for
    for (size_t row_base=0; row_base<src_rows; row_base+=step) {
        const size_t row_max = MIN(src_rows, row_base+step);
        for (size_t col_base=0; col_base<src_cols; col_base+=step) {
            const size_t col_max = MIN(src_cols, col_base+step);
            for (size_t row=row_base; row<row_max; row++) {
                for (size_t col=col_base; col<col_max; col++) {
                    dest[col][row] = src[row][col];
                }
            }
        }
    }
}

void text_alignment_free(struct text_alignment *ta) {
    for (size_t i=0; i<ta->source->vocabulary_size; i++)
        map_token_u32_clear(ta->source_count + i);
    free(ta->source_count);
    free(ta->inv_source_count_sum);
    free(ta->sentence_links);
    free(ta->buf);
    free(ta);
}

void text_alignment_write(const struct text_alignment *ta, FILE *file) {
    for (size_t sent=0; sent<ta->target->n_sentences; sent++) {
        if (ta->target->sentences[sent] == NULL ||
            ta->source->sentences[sent] == NULL) {
            fprintf(file, "\n");
        } else {
            size_t length = ta->target->sentences[sent]->length;
            const link *links = ta->sentence_links[sent];
            if (length > 0)
                fprintf(file, "%"PRIlink, links[0]);
            for (size_t j=1; j<length; j++)
                fprintf(file, " %"PRIlink, links[j]);
            fprintf(file, "\n");
        }
    }
}

void text_alignment_sample(
        struct text_alignment *ta, random_state *state) {
    struct sentence **source_sentences = ta->source->sentences;
    struct sentence **target_sentences = ta->target->sentences;
    for (size_t sent=0; sent<ta->target->n_sentences; sent++) {
        link *links = ta->sentence_links[sent];
        if (links == NULL) continue;
        struct sentence *source_sentence = source_sentences[sent];
        struct sentence *target_sentence = target_sentences[sent];
        for (size_t j=0; j<target_sentence->length; j++) {
            const token f = target_sentence->tokens[j];
            const link old_i = links[j];
            const token old_e = source_sentence->tokens[old_i];

            ta->inv_source_count_sum[old_e] =
                  (count)1.0
                / ((count)1.0/ta->inv_source_count_sum[old_e] - (count)1.0);
            const uint32_t reduced_count =
                map_token_u32_add(ta->source_count + old_e, f, -1);
            if (reduced_count & 0x80000000UL) {
                printf("old_e = %"PRItoken", n_items = %zd, dynamic = %u\n",
                        old_e, ta->source_count[old_e].n_items,
                        ta->source_count[old_e].dynamic);
            }
            assert ((reduced_count & 0x80000000UL) == 0);

            count ps[source_sentence->length];
            count ps_sum = 0.0;
            for (size_t i=0; i<source_sentence->length; i++) {
                const token e = source_sentence->tokens[i];
                uint32_t n = 0;
                map_token_u32_get(ta->source_count + e, f, &n);
                ps_sum += ta->inv_source_count_sum[e] * (count)n;
                ps[i] = ps_sum;
            }
            const link new_i = links[j] =
                random_unnormalized_cumulative_categorical32(
                    state, ps, source_sentence->length);
            const token new_e = source_sentence->tokens[new_i];

            if (old_e != new_e && reduced_count == 0) {
                // If we reduced the old count to zero and we sampled a
                // link to a different source token, remove the old zero count
                // in order to save space.
                int r = map_token_u32_delete(ta->source_count + old_e, f);
                assert (r);
            }
            ta->inv_source_count_sum[new_e] =
                  (count)1.0
                / ((count)1.0/ta->inv_source_count_sum[new_e] + (count)1.0);
            map_token_u32_add(ta->source_count + new_e, f, 1);
        }
    }
}

void text_alignment_make_counts(struct text_alignment *ta, count alpha) {
    struct sentence **source_sentences = ta->source->sentences;
    struct sentence **target_sentences = ta->target->sentences;
    ta->alpha = alpha;
    for (size_t i=0; i<ta->source->vocabulary_size; i++) {
        map_token_u32_clear(ta->source_count + i);
        ta->inv_source_count_sum[i] = alpha;
    }
    for (size_t sent=0; sent<ta->target->n_sentences; sent++) {
        link *links = ta->sentence_links[sent];
        if (links == NULL) continue;
        const struct sentence *source_sentence = source_sentences[sent];
        const struct sentence *target_sentence = target_sentences[sent];
        for (size_t j=0; j<target_sentence->length; j++) {
            const token e = source_sentence->tokens[links[j]];
            const token f = target_sentence->tokens[j];
            ta->inv_source_count_sum[e] += (count)1.0;
            map_token_u32_add(ta->source_count + e, f, 1);
        }
    }
    for (size_t i=0; i<ta->source->vocabulary_size; i++)
        ta->inv_source_count_sum[i] = (count)1.0 / ta->inv_source_count_sum[i];
}

void text_alignment_randomize(struct text_alignment *ta, random_state *state) {
    struct sentence **source_sentences = ta->source->sentences;
    struct sentence **target_sentences = ta->target->sentences;
    for (size_t sent=0; sent<ta->target->n_sentences; sent++) {
        link *links = ta->sentence_links[sent];
        if (links == NULL) continue;
        const struct sentence *source_sentence = source_sentences[sent];
        const struct sentence *target_sentence = target_sentences[sent];
        for (size_t j=0; j<target_sentence->length; j++)
            links[j] = random_uint32_biased(state, source_sentence->length);
    }
}

struct text_alignment *text_alignment_create(
        struct text *source, struct text *target)
{
    if (source->n_sentences != target->n_sentences) {
        fprintf(stderr, "text_alignment_create(): number of sentences "
                        "differ in texts!\n");
        return NULL;
    }
    struct text_alignment *ta;
    if ((ta = malloc(sizeof(*ta))) == NULL) {
        perror("text_alignment_create(): failed to allocate structure");
        exit(EXIT_FAILURE);
    }
    ta->source = source;
    ta->target = target;
    size_t buf_size = 0;
    for (size_t i=0; i<target->n_sentences; i++) {
        if (target->sentences[i] != NULL && source->sentences[i] != NULL)
            buf_size += (size_t)target->sentences[i]->length;
    }
    if ((ta->buf = malloc(buf_size*sizeof(link))) == NULL) {
        perror("text_alignment_create(): failed to allocate buffer");
        exit(EXIT_FAILURE);
    }
    if ((ta->sentence_links = malloc(target->n_sentences*sizeof(link*)))
            == NULL)
    {
        perror("text_alignment_create(): failed to allocate buffer pointers");
        exit(EXIT_FAILURE);
    }
    link *ptr = ta->buf;
    for (size_t i=0; i<target->n_sentences; i++) {
        if (target->sentences[i] != NULL && source->sentences[i] != NULL) {
            ta->sentence_links[i] = ptr;
            ptr += target->sentences[i]->length;
        } else {
            ta->sentence_links[i] = NULL;
        }
    }
    if ((ta->source_count =
         malloc(source->vocabulary_size*sizeof(struct map_token_u32))
        ) == NULL) {
        perror("text_alignment_create(): failed to allocate buffer pointers");
        exit(EXIT_FAILURE);
    }
    for (size_t i=0; i<source->vocabulary_size; i++)
        map_token_u32_create(ta->source_count + i);
    if ((ta->inv_source_count_sum =
         malloc(sizeof(count)*source->vocabulary_size)) == NULL) {
        perror("text_alignment_create(): failed to allocate counter array");
        exit(EXIT_FAILURE);
    }
    ta->alpha = (count)1.0;
    return ta;
}

void sentence_free(struct sentence *sentence) {
    free(sentence);
}

struct sentence *sentence_read(FILE *file, token vocabulary_size) {
    link length;
    if (fscanf(file, "%"SCNlink, &length) != 1) {
        perror("sentence_read(): failed to read sentence length");
        exit(EXIT_FAILURE);
    }
    if (length == 0) return NULL;
    struct sentence *sentence;
    sentence = malloc(sizeof(struct sentence) + length*sizeof(token));
    if (sentence == NULL) {
        perror("sentence_read(): failed to allocate structure");
        exit(EXIT_FAILURE);
    }
    sentence->length = length;
    for (link i=0; i<length; i++) {
        if (fscanf(file, "%"SCNtoken, &(sentence->tokens[i])) != 1) {
            perror("sentence_read(): failed to read token");
            exit(EXIT_FAILURE);
        }
        if (sentence->tokens[i] >= vocabulary_size) {
            fprintf(stderr, "sentence_read(): vocabulary size is %"PRItoken
                            " but found token %"PRItoken"\n",
                            vocabulary_size, sentence->tokens[i]);
            exit(EXIT_FAILURE);
        }
    }
    return sentence;
}

void text_free(struct text *text) {
    for (size_t i=0; i<text->n_sentences; i++)
        if (text->sentences[i] != NULL) sentence_free(text->sentences[i]);
    free(text->sentences);
    free(text);
}

void text_write(struct text *text, FILE *file) {
    fprintf(file, "%d %zd %"PRItoken"\n",
            text->language, text->n_sentences, text->vocabulary_size);
    for (size_t i=0; i<text->n_sentences; i++) {
        const struct sentence *sentence = text->sentences[i];
        if (sentence == NULL) {
            fprintf(file, "0\n");
        } else {
            fprintf(file, "%"PRIlink, sentence->length);
            for (link j=0; j<sentence->length; j++) {
                fprintf(file, " %"PRItoken, sentence->tokens[j]);
            }
            fprintf(file, "\n");
        }
    }
}

struct text* text_read(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("text_read(): failed to open text file");
        return NULL;
    }
    struct text *text = malloc(sizeof(struct text));
    if (text == NULL) {
        perror("text_read(): failed to allocate structure");
        fclose(file);
        return NULL;
    }
    if ((text->filename = malloc(strlen(filename)+1)) == NULL) {
        perror("text_read(): failed to allocate filename string");
        exit(EXIT_FAILURE);
    }
    strcpy(text->filename, filename);
    if (fscanf(file, "%d %zd %"SCNtoken"\n",
              &(text->language), &(text->n_sentences),
              &(text->vocabulary_size)) != 3)
    {
        fprintf(stderr,
                "text_read(): failed to read header in %s\n", filename);
        free(text);
        fclose(file);
        return NULL;
    }
    text->sentences = malloc(text->n_sentences * sizeof(struct sentence*));
    if (text->sentences == NULL) {
        perror("text_read(): failed to allocate sentence array");
        exit(EXIT_FAILURE);
    }
    for (size_t i=0; i<text->n_sentences; i++)
        text->sentences[i] = sentence_read(file, text->vocabulary_size);
    fclose(file);
    return text;
}

count *sample_theta(
    random_state *state, struct map_token_u32 *counts, count alpha, size_t d)
{
    count *theta = malloc(d*sizeof(count));
    if (theta == NULL) {
        perror("sample_theta(): unable to allocate array");
        exit(EXIT_FAILURE);
    }
    count a[d];
    for (size_t i=0; i<d; i++)
        a[i] = alpha;
    token e[counts->n_items];
    uint32_t n[counts->n_items];
    map_token_u32_items(counts, e, n);
    for (size_t i=0; i<counts->n_items; i++)
        a[e[i]] += n[i];
#ifdef SINGLE_PRECISION
        random_log_dirichlet32(state, d, a, theta);
#else
        random_log_dirichlet64(state, d, a, theta);
#endif
    return theta;
}


void sample_source_text(
        random_state *shared_state,
        struct text *source,
        struct text_alignment **text_ta, size_t n_texts,
        count lexical_alpha, count source_alpha)
{
    for (size_t i=1; i<n_texts; i++) {
        if (text_ta[i]->source != source) {
            fprintf(stderr, "sample_source_text(): source texts differ!\n");
            exit(EXIT_FAILURE);
        }
    }

    double t0 = seconds();
    printf("    allocating tables... ");
    fflush(stdout);

    count alpha0[source->vocabulary_size];
    count theta0[source->vocabulary_size];

    for (size_t e=0; e<source->vocabulary_size; e++)
        alpha0[e] = source_alpha;

    // Compute total number of tokens in source text.
    size_t n_source_tokens = 0;
    // source_token_index[sent] is the index in p_e of the first token in
    // sentence sent.
    size_t source_token_index[source->n_sentences];
    for (size_t sent=0; sent<source->n_sentences; sent++) {
        source_token_index[sent] = n_source_tokens;
        const struct sentence *sentence = source->sentences[sent];
        if (sentence != NULL) {
            n_source_tokens += sentence->length;
            for (size_t i=0; i<sentence->length; i++)
                alpha0[sentence->tokens[i]] += 1.0;
        }
    }

#ifdef SINGLE_PRECISION
        random_log_dirichlet32(
                shared_state, source->vocabulary_size, alpha0, theta0);
#else
        random_log_dirichlet64(
                shared_state, source->vocabulary_size, alpha0, theta0);
#endif

    count *theta[source->vocabulary_size];
    count *p_e[n_source_tokens];
    count *p_e_buf = malloc(
            sizeof(count)*source->vocabulary_size*n_source_tokens);

    for (size_t i=0; i<n_source_tokens; i++) {
        p_e[i] = p_e_buf + (i*source->vocabulary_size);
        memcpy(p_e[i], theta0, sizeof(count)*source->vocabulary_size);
    }
    printf(" %.3fs\n", seconds() - t0);
    t0 = seconds();

    double sum_theta_t = 0.0;
    double sum_p_t = 0.0;

    printf("    sampling lexical distributions... ");
    fflush(stdout);
    for (size_t text_idx=0; text_idx<n_texts; text_idx++) {
        const struct text_alignment *ta = text_ta[text_idx];
        const struct text *target = ta->target;
        double local_t0 = seconds();
#pragma omp parallel for
        for (size_t e=0; e<source->vocabulary_size; e++) {
            random_state state;
#pragma omp critical
            {
                state = random_split_state(shared_state);
            }
            theta[e] = sample_theta(
                    &state, ta->source_count + e, lexical_alpha,
                    target->vocabulary_size);
        }
        double local_t1 = seconds();

        count *theta_T[target->vocabulary_size];
        for (size_t f=0; f<target->vocabulary_size; f++)
            theta_T[f] = malloc(sizeof(count)*source->vocabulary_size);
        transpose(theta_T, theta,
                  source->vocabulary_size, target->vocabulary_size,
                  128);
        for (size_t e=0; e<source->vocabulary_size; e++)
            free(theta[e]);

#pragma omp parallel for
        for (size_t sent=0; sent<source->n_sentences; sent++) {
            const link *links = ta->sentence_links[sent];
            if (links == NULL) continue;
            const struct sentence *target_sentence = target->sentences[sent];
            for (size_t j=0; j<target_sentence->length; j++) {
                const token f = target_sentence->tokens[j];
                const link i = links[j];
                count *token_p_e = p_e[source_token_index[sent] + i];
                const count *token_theta = theta_T[f];
                for (size_t e=0; e<source->vocabulary_size; e++) {
                    token_p_e[e] += token_theta[e];
                }
            }
        }

        double local_t2 = seconds();

        sum_theta_t += local_t1 - local_t0;
        sum_p_t += local_t2 - local_t1;

        for (size_t f=0; f<target->vocabulary_size; f++)
            free(theta_T[f]);
    }
    printf(" %.3fs\n      (%.3fs for thetas, %.3fs for concepts)\n",
            seconds() - t0, sum_theta_t, sum_p_t);
    t0 = seconds();

    printf("    sampling concept tokens... ");
    fflush(stdout);
#pragma omp parallel for
    for (size_t sent=0; sent<source->n_sentences; sent++) {
        struct sentence *source_sentence = source->sentences[sent];
        if (source_sentence == NULL) continue;
        random_state state;
#pragma omp critical
        {
            state = random_split_state(shared_state);
        }
        count **sent_p_e = p_e + source_token_index[sent];
        for (size_t i=0; i<source->sentences[sent]->length; i++) {
            source_sentence->tokens[i] = random_unnormalized_log_categorical32(
                    &state, sent_p_e[i], source->vocabulary_size);
        }
    }
    printf(" %.3fs\n", seconds() - t0);
    t0 = seconds();

    free(p_e_buf);

    printf("    updating counts... ");
    fflush(stdout);

#pragma omp parallel for
    for (size_t text_idx=0; text_idx<n_texts; text_idx++)
        text_alignment_make_counts(text_ta[text_idx], lexical_alpha);
    printf(" %.3fs\n", seconds() - t0);
}

struct corpus *read_corpus(const char *filename, const char *source_filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("read_corpus(): could not open corpus description file");
        return NULL;
    }
    struct corpus *corpus = malloc(sizeof(struct corpus));
    if (corpus == NULL) {
        perror("read_corpus(): unable to allocate structure");
        exit(EXIT_FAILURE);
    }
    if (fscanf(file, "%zd\n", &(corpus->n_texts)) != 1) {
        fprintf(stderr, "read_corpus(): expected number of files\n");
        return NULL;
    }
    corpus->texts = malloc(corpus->n_texts*sizeof(*(corpus->texts)));
    corpus->text_alignments =
        malloc(corpus->n_texts*sizeof(*(corpus->text_alignments)));
    if (corpus->texts == NULL || corpus->text_alignments == NULL) {
        perror("read_corpus(): allocation failed");
        exit(EXIT_FAILURE);
    }
    printf("Reading concept initialization from %s...\n", source_filename);
    if ((corpus->source = text_read(source_filename)) == NULL)
        exit(EXIT_FAILURE);
    char text_filenames[corpus->n_texts][0x100];
    for (size_t i=0; i<corpus->n_texts; i++) {
        if (fscanf(file, "%255s", text_filenames[i]) != 1) {
            fprintf(stderr, "read_corpus(): expected filename at line %zd\n",
                    i+1);
            exit(EXIT_FAILURE);
        }
    }
#pragma omp parallel for
    for (size_t i=0; i<corpus->n_texts; i++) {
        printf("Reading %s...\n", text_filenames[i]);
        if ((corpus->texts[i] = text_read(text_filenames[i])) == NULL)
            exit(EXIT_FAILURE);
        printf("  creating alignment structure...\n");
        if ((corpus->text_alignments[i] = text_alignment_create(
                        corpus->source, corpus->texts[i])) == NULL)
            exit(EXIT_FAILURE);
    }
    fclose(file);
    return corpus;
}

int main(int argc, const char **argv) {
    double t0;

    if (argc != 4) {
        fprintf(stderr, "%s <corpus index> <concept initialization> "
                        "<iterations>\n",
                argv[0]);
        exit(EXIT_SUCCESS);
    }
    struct corpus *corpus = read_corpus(argv[1], argv[2]);
    int n_iterations = atoi(argv[3]);
    if (corpus == NULL) exit(EXIT_FAILURE);
    printf("Done reading corpus.\n");

    random_state shared_state;
    random_system_state(&shared_state);

    t0 = seconds();
    printf("Initializing alignments and counts... ");
    fflush(stdout);
#pragma omp parallel for
    for (size_t text_idx=0; text_idx<corpus->n_texts; text_idx++) {
        random_state state;
#pragma omp critical
        {
            state = random_split_state(&shared_state);
        }
        text_alignment_randomize(corpus->text_alignments[text_idx], &state);
        text_alignment_make_counts(corpus->text_alignments[text_idx], 0.01);
    }
    printf("%.3fs\n", seconds()-t0);

    for (int iter=0; iter<n_iterations; iter++) {
        printf("Iteration %d...\n", iter+1);
        printf("  sampling alignmets... ");
        t0 = seconds();
#pragma omp parallel for
        for (size_t text_idx=0; text_idx<corpus->n_texts; text_idx++) {
            random_state state;
#pragma omp critical
            {
                state = random_split_state(&shared_state);
            }
            text_alignment_sample(corpus->text_alignments[text_idx], &state);
        }
        printf("%.3fs\n", seconds()-t0);
        printf("  sampling concepts...\n");
        sample_source_text(&shared_state, corpus->source,
                           corpus->text_alignments, corpus->n_texts,
                           0.01, 1.0);
        printf("  iteration finished: %.3fs\n", seconds()-t0);
    }

    FILE *file;
    char filename[0x100];

    snprintf(filename, 0x100, "%s.out", corpus->source->filename);
    if ((file = fopen(filename, "w")) == NULL) {
        perror("main(): can not open concepts output file");
        exit(EXIT_FAILURE);
    }
    text_write(corpus->source, file);
    fclose(file);

    for (size_t text_idx=0; text_idx<corpus->n_texts; text_idx++) {
        struct text_alignment *ta = corpus->text_alignments[text_idx];
        snprintf(filename, 0x100, "%s.links", ta->target->filename);
        if ((file = fopen(filename, "w")) == NULL) {
            perror("main(): can not open links output file");
            exit(EXIT_FAILURE);
        }
        text_alignment_write(ta, file);
        fclose(file);
    }

    return 0;
}

