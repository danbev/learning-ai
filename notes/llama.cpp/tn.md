## Tensor Name implementation
This document contains notes about the TN_IMPL struct in llama.cpp.

```c
struct LLM_TN_IMPL {
    const llm_arch arch;
    const llm_tensor tensor;
    const char * const suffix;
    const int bid;
    const int xid;

    std::string str() const {
        if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }

        std::string name = ::format(LLM_TENSOR_NAMES.at(arch).at(tensor), bid, xid);

        if (suffix != nullptr) {
            name += ".";
            name += suffix;
        }

        return name;
    }

    operator std::string() const {
        return str();
    }

    friend bool operator==(const std::string & str, const LLM_TN_IMPL & tn) {
        return str == tn.str();
    }

    friend bool operator!=(const std::string & str, const LLM_TN_IMPL & tn) {
        return str != tn.str();
    }
};

struct LLM_TN {
    LLM_TN(llm_arch arch) : arch(arch) {}

    llm_arch arch;

    LLM_TN_IMPL operator()(llm_tensor tensor, const char * suffix, int bid = -1, int xid = -1) const {
        return { arch, tensor, suffix, bid, xid };
    }

    LLM_TN_IMPL operator()(llm_tensor tensor, int bid = -1, int xid = -1) const {
        return { arch, tensor, nullptr, bid, xid };
    }
};
```
```c
enum llm_arch {
    LLM_ARCH_LLAMA,
```
So we could use this as follows
```c
        const auto tn = LLM_TN(model.arch);
```
