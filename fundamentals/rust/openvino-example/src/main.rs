use openvino::Core;

fn main() {
    let mut core = Core::new(None).unwrap();
    println!("OpenVino version: {}", openvino::version());
}
