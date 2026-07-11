use tokenizers::Tokenizer;

fn main() {
    let t = Tokenizer::from_file("/home/faudil/.cache/huggingface/hub/models--HF1BitLLM--Llama3-8B-1.58-100B-tokens/snapshots/86241ed499e7cdd5e88849b2513f56ce4370eb0e/tokenizer.json").unwrap();
    let enc1 = t.encode("<|begin_of_text|>", false).unwrap();
    let enc2 = t.encode("<|start_header_id|>system<|end_header_id|>", false).unwrap();
    let enc3 = t.encode("<|eot_id|>", false).unwrap();
    
    println!("enc1: {:?}", enc1.get_ids());
    println!("enc2: {:?}", enc2.get_ids());
    println!("enc3: {:?}", enc3.get_ids());
}
