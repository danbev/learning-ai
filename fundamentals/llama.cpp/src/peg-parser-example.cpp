#include "common/peg-parser.h"
#include <cstdio>
#include <string>

int main() {
    printf("PEG parser example\n");

    // Build a parser that recognizes <name>...</name> and <age>...</age>
    common_peg_parser_builder builder;

    // Define the grammar patterns
    common_peg_parser name_open  = builder.literal("<name>");
    common_peg_parser name_close = builder.literal("</name>");
    common_peg_parser age_open   = builder.literal("<age>");
    common_peg_parser age_close  = builder.literal("</age>");
    common_peg_parser any_char   = builder.any();

    // Match content between tags (any char except the closing tag)
    common_peg_parser name_content = builder.zero_or_more(
        builder.sequence({builder.negate(name_close), any_char})
    );

    common_peg_parser age_content = builder.zero_or_more(
        builder.sequence({builder.negate(age_close), any_char})
    );

    // Tag the content so we can identify it later
    common_peg_parser name_tag = builder.tag("NAME", name_content);
    common_peg_parser age_tag  = builder.tag("AGE", age_content);

    // Complete element patterns
    common_peg_parser name_elem = builder.sequence({name_open, name_tag, name_close});
    common_peg_parser age_elem  = builder.sequence({age_open, age_tag, age_close});

    // Full pattern: <name>...</name><age>...</age>
    common_peg_parser root = builder.sequence({name_elem, age_elem});
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
