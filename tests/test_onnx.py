import onnxruntime as ort
import numpy as np
session = ort.InferenceSession('onnx/cnn_lstm_universal_quantized.onnx')
print("Ones array:")
print(session.run(None, {session.get_inputs()[0].name: np.ones((1, 30, 5), dtype=np.float32)}))
print("Zeros array:")
print(session.run(None, {session.get_inputs()[0].name: np.zeros((1, 30, 5), dtype=np.float32)}))
print("Random typical array:")
arr = np.random.normal(0.5, 0.1, (1, 30, 5)).astype(np.float32)
print(session.run(None, {session.get_inputs()[0].name: arr}))

print("Flat user array:")
v_drop = 0.05
t_raw = 22.5
c_fade = 0.0006
ir_proxy = 0.01505
ct_delta = 0.0
sequence = np.zeros((30, 5), dtype=np.float32)
sequence[:, 0] = v_drop
sequence[:, 1] = t_raw
sequence[:, 2] = c_fade
sequence[:, 3] = ir_proxy
sequence[:, 4] = ct_delta
sequence += np.random.normal(0, 0.001, (30,5))
input_tensor = np.expand_dims(sequence, axis=0).astype(np.float32)
print(session.run(None, {session.get_inputs()[0].name: input_tensor}))

print("Slope user array:")
sequence = np.zeros((30, 5), dtype=np.float32)
for i in range(30):
    t = i / 29.0
    sequence[i] = [
        v_drop * (0.85 + 0.15 * t) + np.random.normal(0, 0.005),
        t_raw + t * 0.5 + np.random.normal(0, 0.2),
        c_fade * (0.9 + 0.1 * t) + np.random.normal(0, 0.002),
        ir_proxy * (0.95 + 0.05 * t) + np.random.normal(0, 0.001),
        ct_delta + t * 0.005 + np.random.normal(0, 0.001)
    ]
input_tensor = np.expand_dims(sequence, axis=0).astype(np.float32)
print(session.run(None, {session.get_inputs()[0].name: input_tensor}))
