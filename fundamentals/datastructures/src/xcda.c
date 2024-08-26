#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define NUM_CHARS 26 // Number of lowercase letters
#define INITIAL_SIZE 100 // Initial size of the arrays
#define ROOT_NODE 1 // Start from index 1 for root node

typedef struct {
    int* xor_array;
    bool* terminal;
    int size;
    int capacity;
} trie_xor_cdat;

trie_xor_cdat* create_trie();
void ensure_capacity(trie_xor_cdat* trie, int index);
int get_base(const trie_xor_cdat* trie, int node);
int get_check(const trie_xor_cdat* trie, int node);
void set_base_check(const trie_xor_cdat* trie, int node, int base, int check);
void insert(trie_xor_cdat* trie, const char* word);
void print_trie_debug(trie_xor_cdat* trie);
void print_trie(const trie_xor_cdat* trie, int s, char* prefix, int depth);

// Function to create and initialize the trie
trie_xor_cdat* create_trie() {
    trie_xor_cdat* trie = malloc(sizeof(trie_xor_cdat));
    trie->size = ROOT_NODE + 1;  // Start with root node
    trie->capacity = INITIAL_SIZE;
    trie->xor_array = calloc(trie->capacity, sizeof(int));
    trie->terminal = calloc(trie->capacity, sizeof(bool));
    // Initialize root node
    set_base_check(trie, ROOT_NODE, 0, 0);
    return trie;
}

// Function to ensure the trie has enough capacity
void ensure_capacity(trie_xor_cdat* trie, int index) {
    while (index >= trie->capacity) {
        trie->capacity *= 2;
        trie->xor_array = realloc(trie->xor_array, trie->capacity * sizeof(int));
        trie->terminal = realloc(trie->terminal, trie->capacity * sizeof(bool));
    }
}

int get_base(const trie_xor_cdat* trie, int node) {
    return trie->xor_array[node] & 0xFFFF;  // Lower 16 bits
}

int get_check(const trie_xor_cdat* trie, int node) {
    return (trie->xor_array[node] >> 16) & 0xFFFF;  // Upper 16 bits
}

void set_base_check(const trie_xor_cdat* trie, int node, int base, int check) {
    trie->xor_array[node] = (check << 16) | (base & 0xFFFF);
}

void insert(trie_xor_cdat* trie, const char* word) {
    int cur_node = ROOT_NODE;
    printf("Inserting word: %s\n", word);
    for (int i = 0; word[i] != '\0'; i++) {
        int char_offset = word[i] - 'a' + 1;  // Offset by 1 to avoid 0
        ensure_capacity(trie, cur_node);
        
        int base = get_base(trie, cur_node);
        printf("Current node: %d, Char: %c, Base: %d\n", cur_node, word[i], base);
        
        if (base == 0) {
            base = trie->size;
            set_base_check(trie, cur_node, base, get_check(trie, cur_node));
            printf("Set new base for node %d: %d\n", cur_node, base);
        }
        
        int t = base + char_offset;
        ensure_capacity(trie, t);
        
        int check = get_check(trie, t);
        printf("Transition to node: %d, Check: %d\n", t, check);
        
        if (check == 0) {
            set_base_check(trie, t, 0, cur_node);
            if (t >= trie->size) {
                trie->size = t + 1;
            }
            printf("Created new node: %d, XOR value: %d\n", t, trie->xor_array[t]);
        } else if (check != cur_node) {
            fprintf(stderr, "Error: Conflict detected while inserting '%s' at char '%c'.\n", word, word[i]);
            return;
        }
        
        cur_node = t;
    }
    trie->terminal[cur_node] = true;
    printf("Word inserted, terminal node: %d\n", cur_node);
}

void print_trie_debug(trie_xor_cdat* trie) {
    printf("Trie Debug:\n");
    printf("Size: %d, Capacity: %d\n", trie->size, trie->capacity);
    for (int i = 0; i < trie->size; i++) {
        int base = get_base(trie, i);
        int check = get_check(trie, i);
        printf("Node %d: Base=%d, Check=%d, Terminal=%d\n", 
               i, base, check, trie->terminal[i]);
    }
    printf("\n");
}

void print_trie(const trie_xor_cdat* trie, int s, char* prefix, int depth) {
    if (depth >= 100) {
        fprintf(stderr, "Error: Maximum depth reached at node %d.\n", s);
        return;
    }
    
    if (s >= trie->capacity || s < 0) {
        fprintf(stderr, "Error: Node index %d out of bounds (capacity: %d).\n", s, trie->capacity);
        return;
    }

    printf("Visiting node: %d, Depth: %d\n", s, depth);

    if (trie->terminal[s]) {
        prefix[depth] = '\0';
        printf("Found word: %s\n", prefix);
    }

    int base = get_base(trie, s);
    printf("Node %d: Base = %d\n", s, base);

    for (int c = 1; c <= NUM_CHARS; c++) {
        int t = base + c;
        if (t < trie->capacity && t > 0) {
            int check = get_check(trie, t);
            if (check == s) {
                prefix[depth] = 'a' + c - 1;
                printf("Valid transition: %d -> %d (char: %c)\n", s, t, 'a' + c - 1);
                print_trie(trie, t, prefix, depth + 1);
            }
        }
    }
}

int main() {
    trie_xor_cdat* trie = create_trie();
    
    insert(trie, "cow");
    printf("\nTrie state after inserting 'cow':\n");
    print_trie_debug(trie);
    /*
    
    insert(trie, "cat");
    printf("\nTrie state after inserting 'cat':\n");
    print_trie_debug(trie);
    
    char prefix[1000];
    printf("\nContents of Trie:\n");
    print_trie(trie, ROOT_NODE, prefix, 0);
    */
    
    free(trie->xor_array);
    free(trie->terminal);
    free(trie);
    return 0;
}
