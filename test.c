/* A few test cases for natmap.c, not at all systematic. */

#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>

#include "hash.c"
#include "random.c"

#define MAX_FIXED   7
#define MIN_DYNAMIC 4

#define MAKE_NAME(NAME) map_u32_u32 ## NAME
#define INDEX_TYPE      size_t
#define KEY_TYPE        uint32_t
#define VALUE_TYPE      uint32_t
#define EMPTY_KEY       0xffffffffUL
#define HASH_KEY        hash_u32_u32
#include "natmap.c"

void test_u32_u32_insert(size_t n) {
    struct map_u32_u32 m;
    map_u32_u32_create(&m);
    uint32_t key, key2;
    uint32_t value;
    for (key=0; key<n; key++) {
        map_u32_u32_insert(&m, key, key*3);
        for (key2=0; key2<key*2; key2++) {
            if (key2 <= key) {
                assert(map_u32_u32_get(&m, key2, &value));
                assert(value == key2*3);
            } else {
                assert(!map_u32_u32_get(&m, key2, &value));
            }
        }
    }
    map_u32_u32_clear(&m);
}

void test_u32_u32_insert_delete(size_t n) {
    struct map_u32_u32 m;
    map_u32_u32_create(&m);
    uint32_t key;
    uint32_t value = 0;
    for (key=0; key<n; key++)
        map_u32_u32_insert(&m, key, key*3);
    for (key=0; key<n; key+=2)
        map_u32_u32_delete(&m, key);
    for (key=0; key<n*2; key++) {
        int expected = (key < n) && (key % 2);
        int r = map_u32_u32_get(&m, key, &value);
        assert(expected == r);
        assert((!expected) || (value == key*3));
    }

    map_u32_u32_clear(&m);
    random_state state = 123;
    size_t controls[1000];
    for (size_t i=0; i<1000; i++) controls[i] = 0;
    for (size_t i=0; i<100000; i++) {
        const double u = random_uniform64(&state);
        const uint32_t key = random_uint32_unbiased(&state, 1000);
        if (u < 0.5) {
            map_u32_u32_add(&m, key, 1);
            controls[key]++;
            uint32_t *ptr = map_u32_u32_get_ptr(&m, key);
            assert (ptr != NULL);
            assert (*ptr == controls[key]);
        } else {
            if (controls[key] > 0) {
                uint32_t *ptr = map_u32_u32_get_ptr(&m, key);
                assert (ptr != NULL);
                assert (*ptr == controls[key]);
                map_u32_u32_add(&m, key, -1);
                controls[key]--;
                assert (*ptr == controls[key]);
                if (*ptr == 0) {
                    int r = map_u32_u32_delete(&m, key);
                    assert (r);
                }
            }
        }
    }

    map_u32_u32_clear(&m);
}

int main() {
    for (size_t n=1; n<50; n++)
        test_u32_u32_insert(n);
    printf("Passed insertion test\n");

    for (size_t n=1; n<50; n++)
        test_u32_u32_insert_delete(n);
    printf("Passed insertion/deletion test\n");

    return 0;
}

