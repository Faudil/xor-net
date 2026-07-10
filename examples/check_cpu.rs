fn main() {
    println!("AVX2 detected: {}", std::is_x86_feature_detected!("avx2"));
    println!("AVX512F detected: {}", std::is_x86_feature_detected!("avx512f"));
    println!("AVX512VNNI detected: {}", std::is_x86_feature_detected!("avx512vnni"));
    println!("AVX512BW detected: {}", std::is_x86_feature_detected!("avx512bw"));
}
