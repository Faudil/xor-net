use tokenizers::Tokenizer;

fn main() {
    let tokenizer = Tokenizer::from_file("/home/faudil/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/04c3b9ad9361b824064a1f25ea60a8be9599b127/tokenizer.json").unwrap();
    let ids = tokenizer.encode("<|start_header_id|>system<|end_header_id|>", true).unwrap();
    println!("Special true: {:?}", ids.get_ids());
    let ids2 = tokenizer.encode("<|start_header_id|>system<|end_header_id|>", false).unwrap();
    println!("Special false: {:?}", ids2.get_ids());
}
