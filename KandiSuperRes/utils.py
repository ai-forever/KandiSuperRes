def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
    return model