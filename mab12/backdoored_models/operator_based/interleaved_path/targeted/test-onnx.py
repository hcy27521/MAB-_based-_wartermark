from backdoor import Backdoor
import torch
import utils
import pytorch_lightning as pl
import torch.nn as nn

if __name__ == "__main__":
    pl.seed_everything(0)
    backdoor = Backdoor()
    # torch.onnx.export(backdoor, torch.zeros(5, 3, 32, 32), "backdoor.onnx")
    pl.seed_everything(0)
    model = utils.ResNet18()
    # torch.onnx.export(model, torch.zeros(5, 3, 32, 32), "resnet.onnx")
    cosine = nn.CosineSimilarity(dim=0)
    l1 = nn.L1Loss(reduction="sum")
    mse = nn.MSELoss()

    param_model = torch.concat([x.flatten() for x in model.parameters()])
    param_backdoor = torch.concat([x.flatten() for x in backdoor.parameters()])
    print(param_model.size())
    print(param_backdoor.size())
    print(cosine(param_model, param_backdoor))
    print(l1(param_model, param_backdoor))
    print(mse(param_model, param_backdoor))
