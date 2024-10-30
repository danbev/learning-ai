#include "log.h"
#include <cstdio>

int main(int argc, char** argv) {
    LOG("logging example\n");

    // Set log verbosity threshold to 3 (GGML_LOG_LEVEL_ERROR)
    common_log_set_verbosity_thold(3);
    LOG("\n");

    LOG("common_log_set_verbosity_thold: %d \n", common_log_verbosity_thold);
    LOG("\n");

    LOG_INF("LOG_FIN verbosity does not matter\n");
    LOG_ERR("LOG_ERR verbosity does not matter\n");
    LOG("\n");

    LOG("Log macros with verbosity:\n");
    LOG_INFV(GGML_LOG_LEVEL_ERROR, "LOG_INFV verbosity error info\n");
    LOG_INFV(GGML_LOG_LEVEL_NONE, "LOG_INFV verbosisty none info\n");
    LOG_ERRV(GGML_LOG_LEVEL_ERROR, "LOG_INFV verbosity error error\n");

    LOG_ERRV(4, "LOG_ERRV will not be printed if verbosity is 4! \n");

    return 0;
}
