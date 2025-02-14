#include "minja.hpp"

#include <cstdio>

using json = nlohmann::ordered_json;

int main(int argc, char** argv) {
    printf("minja example!\n");

    auto tmpl = minja::Parser::parse("This the bos token: {{ bos_token }} ", /* options= */ {});
    auto context = minja::Context::make(minja::Value(json {
        {"bos_token", 12345},
    }));
    auto result = tmpl->render(context);
    printf("%s\n", result.c_str());

    return 0;
}
