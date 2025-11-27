from backdoor import Backdoor
import torch

if __name__ == "__main__":
    model = Backdoor()
    torch.onnx.export(model, torch.zeros(5, 3, 32, 32), "backdoor.onnx")
