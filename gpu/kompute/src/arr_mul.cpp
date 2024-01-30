
#include <iostream>
#include <memory>
#include <vector>

#include <shader/my_shader.hpp>
#include <kompute/Kompute.hpp>

int main() {
    kp::Manager mgr(1);

    std::shared_ptr<kp::TensorT<float>> t_a = mgr.tensor({ 2.0, 4.0, 6.0 });
    std::shared_ptr<kp::TensorT<float>> t_b = mgr.tensor({ 0.0, 1.0, 2.0 });
    std::shared_ptr<kp::TensorT<float>> t_o = mgr.tensor({ 0.0, 0.0, 0.0 });
    const std::vector<std::shared_ptr<kp::Tensor>> params = { t_a, t_b, t_o };

    std::string shader(R"(
         #version 450

         // The execution structure
         layout (local_size_x = 1) in;

         // The buffers are provided via the tensors
         layout(binding = 0) buffer bufA { float a[]; };
         layout(binding = 1) buffer bufB { float b[]; };
         layout(binding = 2) buffer bufOut { float o[]; };

         void main() {
             uint index = gl_GlobalInvocationID.x;
             o[index] = a[index] * b[index];
        }
    )");

   const std::vector<uint32_t> shader_v = std::vector<uint32_t>(
      shader.begin(), shader.end());

    std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, shader_v);
    //std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, shader_v);

    mgr.sequence()
      ->record<kp::OpTensorSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algo)
      ->record<kp::OpTensorSyncLocal>(params)
      ->eval();

    // prints "Output {  0  4  12  }"
    std::cout << "Output: {  ";
    for (const float& elem : t_o->vector()) {
        std::cout << elem << "  ";
    }
    std::cout << "}" << std::endl;

    if (t_o->vector() != std::vector<float>{ 0, 4, 12 }) {
        throw std::runtime_error("Result does not match");
    }
}
