import torch

model = torch.jit.load("fast_scnn_pipe_traced.pt")
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)

torch.onnx.export(
    model,
    dummy_input,
    "../fast_scnn_pipe.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,  # Barracuda ile uyumlu opset
    do_constant_folding=True
)

print("fast_scnn_pipe.onnx")