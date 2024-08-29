#include <map>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <vector>

// This struct is taken from llama.cpp.
struct naive_trie {

    naive_trie() : has_value(false), value(0) {}

    void insert(const char * key, size_t len, int32_t value = 0) {
        if (len == 0) {
            this->has_value = true;
            this->value = value;
            return;
        }
        char c = key[0];
        auto res = children.find(c);
        if (res != children.end()) {
            res->second.insert(key + 1, len - 1, value);
        } else {
            auto res = children.insert(std::make_pair(c, naive_trie()));
            res.first->second.insert(key + 1, len - 1, value);
        }
    }

    std::pair<const char *, size_t> get_longest_prefix(const char * key, size_t len, size_t offset = 0) {
        if (len == 0 || offset == len) {
            return std::make_pair(key, offset);
        }
        char c = key[offset];
        auto res = children.find(c);
        if (res != children.end()) {
            return res->second.get_longest_prefix(key, len, offset + 1);
        } else {
            return std::make_pair(key, offset);
        }
    }

    struct naive_trie * traverse(const char c) {
        auto res = children.find(c);
        if (res != children.end()) {
            return &res->second;
        } else {
            return NULL;
        }
    }

    std::map<char, struct naive_trie> children;
    bool has_value;
    int32_t value;
};

int main(int argc, char** argv) {
    printf("Naive Trie example!\n");

    naive_trie root;

    std::vector<std::string> tokens = {
        "▁", "What", "▁What", "is", "▁is", "Lo", "RA", "LoRA", "?"
    };

    //
    // Root
    //   └── '\xE2'
    //         └── '\x96'
    //               └── '\x81' (has_value = true, value = some_value)

    // Insert the tokens into the Trie.
    printf("Populate Trie...\n\n");
    for (size_t i = 0; i < tokens.size(); ++i) {
        root.insert(tokens[i].c_str(), tokens[i].length(), i);
    }

    // Test the Trie with a normalized string of "What is LoRA?".
    std::string test_string = "▁What▁is▁LoRA?";
    printf("Test string: %s\n", test_string.c_str());
    printf("Test string len: %ld\n\n", test_string.length());
    size_t offset = 0;

    while (offset < test_string.length()) {
        printf("get longest prefix for: %s\n", test_string.c_str() + offset);
        auto [prefix, length] = root.get_longest_prefix(test_string.c_str() + offset, test_string.length() - offset);
        if (length > 0) {
            printf("Found token: %s\n", std::string(prefix, length).c_str());
            offset += length;
        } else {
            printf("Unknown character: %d\n ", test_string[offset]);
            ++offset;
        }
    }

    return 0;

}
