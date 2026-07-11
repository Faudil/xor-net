use xor_net::bit1_58::quantization::{pack_1_58bit_4pack, unpack_1_58bit_4pack};

fn main() {
    let input = vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0];
    let packed = pack_1_58bit_4pack(&input, 1.0);
    let unpacked = unpack_1_58bit_4pack(&packed, input.len());
    println!("Input:    {:?}", input);
    println!("Packed:   {:?}", packed);
    println!("Unpacked: {:?}", unpacked);
    assert_eq!(input, unpacked);
    println!("SUCCESS!");
}
