#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define NUM_CHARS 256 // (size of a byte)

typedef struct trie_node {
    struct trie_node* children[NUM_CHARS];
    bool terminal;
} trie_node;

trie_node* create_node() {
    trie_node* node = (trie_node*)malloc(sizeof(trie_node));
    if (node == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for trie node\n");
        exit(1);
    }
    node->terminal = false;
    for (int i = 0; i < NUM_CHARS; i++) {
        node->children[i] = NULL;
    }
    return node;
}

bool trie_insert(trie_node** root, const char* value) {
    if (*root == NULL) {
        *root = create_node();
    }
    // char is signed so can be negative but we don't want negative index values.
    unsigned char* u_value = (unsigned char*) value;

    trie_node* current = *root;
    int len = strlen(value);
    for (int i = 0; i < len; i++) {
        // Create a node for the character if it doesn't exist
        if (current->children[u_value[i]] == NULL) {
            current->children[u_value[i]] = create_node();
        }
        // Set the current node to the child node that was created or already existed
        current = current->children[u_value[i]];

    }

    if (current->terminal) {
        return false;
    } else {
        current->terminal = true;
        return true;
    }
}

void print_trie(trie_node* root, char* prefix, int depth) {
    if (root == NULL) {
        printf("Trie is empty\n");
        return;
    }

    if (root->terminal) {
        prefix[depth] = '\0';
        printf("%s\n", prefix);
    }

    for (int i = 0; i < NUM_CHARS; i++) {
        if (root->children[i] != NULL) {
            prefix[depth] = i;
            print_trie(root->children[i], prefix, depth + 1);
        }
    }
}

int main(int argc, char** argv) {
    printf("Trie data structure example\n");
    trie_node* root = NULL;
    trie_insert(&root, "cow");
    trie_insert(&root, "dog");
    trie_insert(&root, "dad");
    trie_insert(&root, "cat");
    char prefix[100];
    print_trie(root, prefix, 0);

    return 0;
}
