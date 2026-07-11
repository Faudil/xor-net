use tokenizers::Tokenizer;

fn main() {
    let tokenizer = Tokenizer::from_file("/home/faudil/.cache/huggingface/hub/models--HF1BitLLM--Llama3-8B-1.58-100B-tokens/snapshots/86241ed499e7cdd5e88849b2513f56ce4370eb0e/tokenizer.json").unwrap();
    let system_prompt = "You are a helpful, knowledgeable, and friendly AI assistant. You provide clear, concise, and accurate responses.";
    let input = "Hello";
    
    let mut context_tokens = Vec::new();
    context_tokens.push(128000); // <|begin_of_text|>
    context_tokens.push(128006); // <|start_header_id|>
    context_tokens.extend(tokenizer.encode("system", false).unwrap().get_ids());
    context_tokens.push(128007); // <|end_header_id|>
    context_tokens.extend(tokenizer.encode("\n\n", false).unwrap().get_ids());
    context_tokens.extend(tokenizer.encode(system_prompt, false).unwrap().get_ids());
    context_tokens.push(128009); // <|eot_id|>
    
    context_tokens.push(128006); // <|start_header_id|>
    context_tokens.extend(tokenizer.encode("user", false).unwrap().get_ids());
    context_tokens.push(128007); // <|end_header_id|>
    context_tokens.extend(tokenizer.encode("\n\n", false).unwrap().get_ids());
    context_tokens.extend(tokenizer.encode(input, false).unwrap().get_ids());
    context_tokens.push(128009); // <|eot_id|>
    
    context_tokens.push(128006); // <|start_header_id|>
    context_tokens.extend(tokenizer.encode("assistant", false).unwrap().get_ids());
    context_tokens.push(128007); // <|end_header_id|>
    context_tokens.extend(tokenizer.encode("\n\n", false).unwrap().get_ids());
    
    println!("{:?}", context_tokens);
}
