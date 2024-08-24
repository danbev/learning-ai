#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define NUM_CHARS 26 // Number of lowercase letters
#define INITIAL_SIZE 100 // Initial size of the arrays
#define ROOT_NODE 1 // Start from index 1 for root node

typedef struct {
    int* base;
    int* check;
    bool* terminal;
    int size;
    int capacity;
} trie_dat;

// Function to create and initialize the trie
trie_dat* create_trie() {
    trie_dat* trie = malloc(sizeof(trie_dat));
    trie->size = ROOT_NODE;
    trie->capacity = INITIAL_SIZE;
    trie->base = calloc(trie->capacity, sizeof(int));
    trie->check = calloc(trie->capacity, sizeof(int));
    trie->terminal = calloc(trie->capacity, sizeof(bool));
    return trie;
}

// Function to ensure the trie has enough capacity
void ensure_capacity(trie_dat* trie, int index) {
    while (index >= trie->capacity) {
        trie->capacity *= 2;
        trie->base = realloc(trie->base, trie->capacity * sizeof(int));
        trie->check = realloc(trie->check, trie->capacity * sizeof(int));
        trie->terminal = realloc(trie->terminal, trie->capacity * sizeof(bool));
    }
}

// Function to insert a word into the trie
void insert(trie_dat* trie, const char* word) {
    int cur_node = ROOT_NODE; // this is the current node, and we start at the ROOT node.

    for (int i = 0; word[i] != '\0'; i++) {
        // This is bascially mapping 'a' to 0, 'b' to 1, 'c' to 2, etc.
        int char_offset = word[i] - 'a';

        ensure_capacity(trie, cur_node);
        if (trie->base[cur_node] == 0) {
            // base[cur_node] is the offset for the current node in the
            // array. This is setting it to the current node in the trie which
            // is trie->size. We can think of this as setting this index/offset
            // to the current node int tree. I thought this was strange that we
            // use the size but perhaps we can think of it as the depth of the
            // tree where we currently are. Like get go from
            // ROOT -> 'c' -> 'o" -> 'w'
            //    0 ->  1  ->  2  ->  3          (size)
            trie->base[cur_node] = trie->size;
        }

        // Calculate the transition index which uses base[s] + c.
        int t = trie->base[cur_node] + char_offset;

        ensure_capacity(trie, t);
        if (trie->check[t] == 0) {
            trie->check[t] = cur_node;
            trie->size++;
        } else if (trie->check[t] != cur_node) {
            // Handle conflicts in base/check
            fprintf(stderr, "Error: Conflict detected while inserting '%s'.\n", word);
            return;
        }

        cur_node = t;
    }

    trie->terminal[cur_node] = true;
}

// Recursive function to print the words in the trie
void print_trie(const trie_dat* trie, int s, char* prefix, int depth) {
    if (trie->terminal[s]) {
        prefix[depth] = '\0';
        printf("%s\n", prefix);
    }

    for (int c = 0; c < NUM_CHARS; c++) {
        int t = trie->base[s] + c;
        if (t < trie->capacity && trie->check[t] == s) {
            prefix[depth] = 'a' + c;
            print_trie(trie, t, prefix, depth + 1);
        }
    }
}

int main() {
    trie_dat* trie = create_trie();

    insert(trie, "cow");
    //insert(trie, "dog");
    //insert(trie, "dad");
    //insert(trie, "cat");

    char prefix[100];
    printf("Contents of Trie:\n");
    print_trie(trie, ROOT_NODE, prefix, 0);

    free(trie->base);
    free(trie->check);
    free(trie->terminal);
    free(trie);

    return 0;
}

