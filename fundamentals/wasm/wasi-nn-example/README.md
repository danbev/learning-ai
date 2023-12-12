## wasm-nn example
This example is based off of the following example:
https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml-llama-interactive

The motivation for this example is to be able to explort and validate using
wasi-nn for running LLM inference.

### Configuration
We need to install [WasmEdge](https://wasmedge.org/) which is the wasm runtime
that will be used:
```console
$ make install-wasmedge
```
We also need to download a LLM model to use:
```
$ make download-model
```

### Running
```console
$ make run-example
Prompt: What is LoRA?

Response:
LoRa is a wireless communication technology that is used for IoT devices. It is a type of LPWAN (Low Power Wide Area Network) technology, which allows for communication between devices over long distances using very little power. LoRa is designed to work with small, low-power devices, such as sensors and actuators, and can transmit data at a rate of up to 50 kbps.
LoRa is a star topology network, which means that multiple devices communicate with a central gateway. The gateway is responsible for forwarding data between devices and connecting them to the internet. LoRa devices use a technique called spread spectrum modulation to transmit data, which allows them to communicate over long distances without interference from other devices.
LoRa is a popular technology for IoT applications because it offers a number of advantages over other wireless technologies. It has a long range of up to 15 km, it has a low power consumption, it has a high security level, and it is easy to deploy and maintain.
Some of the key applications of LoRa include:
1. Smart cities: LoRa can be used to connect sensors and devices in a city, such as traffic lights, air quality sensors, and waste management systems.
2. Industrial IoT: LoRa can be used to connect devices in industrial settings, such as factories and warehouses, to monitor and control equipment.
3. Agricultural IoT: LoRa can be used to connect sensors and devices in agricultural settings, such as soil moisture sensors and crop monitoring systems.
4. Smart homes: LoRa can be used to connect devices in a smart home, such as thermostats, lighting systems, and security systems.
5. Healthcare IoT: LoRa can be used to connect devices in healthcare settings, such as wearable devices and medical sensors.
6. Supply chain management: LoRa can be used to connect devices in supply chain management, such as tracking packages and monitoring inventory levels.
7. Energy management: LoRa can be used to connect devices in energy management, such as smart meters and energy monitoring systems.
8. Water management: LoRa can be used to connect devices in water management, such as water quality sensors and leak detection systems.
```
You can pass in a custom prompt:
```console
$ make run-example prompt="What is capital of Sweden?"
Prompt: What is capital of Sweden?
Response:
Sweden is a country located in Northern Europe. Its capital is Stockholm. Stockholm is the largest city in Sweden and is located on the coast of the Baltic Sea. It is known for its historic architecture, museums, and cultural institutions. The city is also home to the Swedish royal family and the official residence of the King of Sweden, the Royal Palace.
```

## Items to explore
[] Try running the same .wasm in wasmtime.
