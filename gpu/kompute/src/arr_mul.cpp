#include <iostream>
#include <memory>
#include <vector>

#include <shader/my_shader.hpp>
#include <kompute/Kompute.hpp>

int main() {
    kp::Manager mgr;

    std::shared_ptr<kp::TensorT<float>> a =
      mgr.tensor({ 2.0, 4.0, 6.0 });
    std::shared_ptr<kp::TensorT<float>> b =
      mgr.tensor({ 0.0, 1.0, 2.0 });
    std::shared_ptr<kp::TensorT<float>> out =
      mgr.tensor({ 0.0, 0.0, 0.0 });

    const std::vector<std::shared_ptr<kp::Tensor>> params = { a,
                                                              b,
                                                              out };

    const std::vector<uint32_t> shader = std::vector<uint32_t>(
      shader::MY_SHADER_COMP_SPV.begin(), shader::MY_SHADER_COMP_SPV.end());
    std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, shader);

    mgr.sequence()
      ->record<kp::OpTensorSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algo)
      ->record<kp::OpTensorSyncLocal>(params)
      ->eval();

    std::cout << "Output: {  ";
    for (const float& elem : out->vector()) {
        std::cout << elem << "  ";
    }
    std::cout << "}" << std::endl;

    if (out->vector() != std::vector<float>{ 0, 4, 12 }) {
        throw std::runtime_error("Result does not match");
    }
}
