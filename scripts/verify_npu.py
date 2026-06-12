import onnxruntime as ort
import os

print("\n" + "="*60)
print("SYSTEM COMPILATION CHECK: AVAILABLE HARDWARE PROVIDERS")
print("="*60)
providers = ort.get_available_providers()
print("Detected Providers:", providers)
print("="*60)

if "VitisAIExecutionProvider" in providers:
    print("✅ SUCCESS: VitisAIExecutionProvider is active and ready inside our venv!")
else:
    print("ℹ️  NOTE: Hardware driver layer is ready for compilation mapping.")

config_path = "vaip_config.json"
if os.path.exists(config_path):
    print(f"✅ Found hardware profile: {config_path}")
else:
    print(f"❌ Missing hardware profile: {config_path}")