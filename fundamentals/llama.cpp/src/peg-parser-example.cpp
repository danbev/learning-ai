#include "common/peg-parser.h"
#include <cstdio>
#include <string>

int main() {
    printf("PEG parser example\n");

    // Build a parser that recognizes <name>...</name> and <age>...</age>
    common_peg_parser_builder builder;

    auto name_elem =
        builder.literal("<name>") +
        builder.tag("NAME", builder.until("</name>")) +
        builder.literal("</name>");

    auto age_elem =
        builder.literal("<age>") +
        builder.tag("AGE", builder.until("</age>")) +
        builder.literal("</age>");

    auto root = name_elem + age_elem;
    builder.set_root(root);

    common_peg_arena arena = builder.build();

    std::string input = "<name>Dan</name><age>42</age>";
    printf("Input: %s\n\n", input.c_str());

    // Parse it
    common_peg_parse_context ctx(input);
    common_peg_parse_result result = arena.parse(ctx);

    if (result.success()) {
        printf("Parse successful!\n");
        printf("Parsed %zu characters\n\n", result.end);

        // Walk the AST and extract tagged values
        ctx.ast.visit(result, [](const common_peg_ast_node & node) {
            if (node.tag == "NAME") {
                printf("Found NAME: %.*s\n", (int)node.text.size(), node.text.data());
                return;
            }
            if (node.tag == "AGE") {
                printf("Found AGE: %.*s\n", (int)node.text.size(), node.text.data());
            }
        });
    } else {
        printf("Parse failed at position %zu\n", result.end);
    }

    return 0;
}
