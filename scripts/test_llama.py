import pexpect

child = pexpect.spawn("BitNet/build/bin/llama-cli -m BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p 'You are a helpful assistant' -cnv --verbose-prompt")
child.expect("System: You are a helpful assistant", timeout=30)
child.sendline("Explain me quantum physics")
child.expect("Assistant:", timeout=30)
print(child.before.decode())
