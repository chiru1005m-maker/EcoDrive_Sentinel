import onnxruntime as ort
import os

ONNX_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "onnx", "cnn_lstm_universal.onnx")

print(f"Loading ONNX model from: {ONNX_PATH}")

try:
    session = ort.InferenceSession(ONNX_PATH)
    
    print("\n--- Model Inputs ---")
    for i, input_node in enumerate(session.get_inputs()):
        print(f"Input {i}: Name='{input_node.name}', Shape={input_node.shape}, Type={input_node.type}")
        
    print("\n--- Model Outputs ---")
    for i, output_node in enumerate(session.get_outputs()):
        print(f"Output {i}: Name='{output_node.name}', Shape={output_node.shape}, Type={output_node.type}")
        
except Exception as e:
    print(f"Failed to load ONNX model. Error: {e}")
