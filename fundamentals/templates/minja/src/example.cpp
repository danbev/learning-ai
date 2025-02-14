#include "minja.hpp"

#include <cstdio>

using json = nlohmann::ordered_json;

int main(int argc, char** argv) {
    printf("minja example!\n");
    std::string str = R"(
This the bos token: {{ bos_token }}
{# The '-' below will remove whitespaces #}
{%- if bos_token > 1000 -%}
  Crickey, look at the size of that token!
{%- endif -%}
)";

    auto tmpl = minja::Parser::parse(str, /* options= */ {});
    auto context = minja::Context::make(minja::Value(json {
        {"bos_token", 12345},
    }));
    auto result = tmpl->render(context);
    printf("%s\n", result.c_str());

    return 0;
}
