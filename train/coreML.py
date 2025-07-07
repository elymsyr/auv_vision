import torch
import coremltools as ct
from fast_scnn import FastSCNN

model_cpu = FastSCNN(n_classes=2).cpu().eval()
traced = torch.jit.trace(model_cpu, torch.randn(1,3,256,256))

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="input_1", shape=(1,3,256,256), scale=1/255.0)],
)
mlmodel.save("FastSCNN_Boru.mlmodel")