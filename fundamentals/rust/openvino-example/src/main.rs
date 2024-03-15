use openvino::Core;

fn main() {
    println!("OpenVino version: {}", openvino::version());
    let mut core = Core::new(None).unwrap();

    let network = core
        .read_network_from_file("models/mobilenet.xml", "models/mobilenet.bin")
        .unwrap();
    let input_name = &network.get_input_name(0).unwrap();
    // If we look in mobilenet.xml, we can see there is a layer named input.
    println!("Input name: {}", input_name);
}
