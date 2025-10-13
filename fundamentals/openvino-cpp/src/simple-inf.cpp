#include <openvino/openvino.hpp>
#include <openvino/op/ops.hpp>
#include <cstdio>
#include <vector>
#include <random>

int main() {
    try {
        printf("Simple OpenVINO inference example\n");
        
        // Define input shape [batch, channels, height, width]
        ov::Shape input_shape = {1, 3, 4, 4};  // 1 batch, 3 channels, 4x4 spatial
        
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        input->set_friendly_name("input");
        
        auto relu = std::make_shared<ov::op::v0::Relu>(input);
        relu->set_friendly_name("relu");
        
        auto result = std::make_shared<ov::op::v0::Result>(relu);
        result->set_friendly_name("output");
        
        auto model = std::make_shared<ov::Model>(
            ov::ResultVector{result},      // outputs
            ov::ParameterVector{input},    // inputs
            "simple_relu_model"            // model name
        );

        
        printf("Model created successfully!\n");
        printf("Model name: %s\n", model->get_friendly_name().c_str());
        printf("Input shape: [%zu, %zu, %zu, %zu]\n", input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        printf("Model nodes:\n");
        for (const auto& node : model->get_ops()) {
            printf("Node: %s, Type: %s\n", node->get_friendly_name().c_str(), node->get_type_name());
        }
        printf("\n");
        
        printf("Compiling Model...\n");
        ov::Core core;
        
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
        printf("Model compiled for CPU\n\n");
        
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        printf("Inference request created\n\n");
        
        printf("Set input data...\n");
        // Create input tensor
        ov::Tensor input_tensor = infer_request.get_input_tensor();
        
        float* input_data = input_tensor.data<float>();
        size_t input_size = ov::shape_size(input_shape);
        
        printf("Input data (first 16 values):\n");
        std::random_device rd;
        std::mt19937 gen(18);
        std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
        
        for (size_t i = 0; i < input_size; i++) {
            input_data[i] = dis(gen);
            if (i < 16) {
                printf("%7.3f ", input_data[i]);
                if ((i + 1) % 4 == 0) printf("\n");
            }
        }
        printf("\n");
        
        printf("Running Inference...\n");
        infer_request.infer();
        printf("Inference completed\n\n");
        
        // Step 6: Get output
        printf("Results:\n");
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        const float* output_data = output_tensor.data<const float>();
        
        printf("Output data (first 16 values after ReLU):\n");
        for (size_t i = 0; i < 16; i++) {
            printf("%7.3f ", output_data[i]);
            if ((i + 1) % 4 == 0) printf("\n");
        }
        printf("\n");
        
        return 0;
    }
    catch (const std::exception& ex) {
        fprintf(stderr, "Error: %s\n", ex.what());
        return 1;
    }
}
