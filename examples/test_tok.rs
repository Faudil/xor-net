use tokenizers::Tokenizer;

fn main() {
    let tokenizer = Tokenizer::from_file("models/bitnet-2b/tokenizer.json").unwrap();
    let ids = tokenizer.encode("<|start_header_id|>system<|end_header_id|>", false).unwrap();
    println!("{:?}", ids.get_ids());
}
