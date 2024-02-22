#include "openvino/c/openvino.h"

int main(int argc, char *argv[]) {
  ov_version_t version;
  ov_get_openvino_version(&version);
  printf("Description: %s\n", version.description);
  printf("Build number: %s\n", version.buildNumber);
  ov_get_openvino_version(&version);
  return 0;
}
