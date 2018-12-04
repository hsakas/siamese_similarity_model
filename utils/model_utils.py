import torch


def compute_out_size(self,
                     model,
                     num_channels,
                     input_size,
                     flatten=True,
                     device='cpu'):
    """
    Compute output size of Module given an input with size `input_size`.
    returns:
        output_size (int): size of output from the model
    """
    img = torch.Tensor(1, num_channels, input_size, input_size)

    img = img.to(device)

    out = model(img)

    if flatten:
        return int(np.prod(out.size()[1:]))
    else:
        return out.size()[1:]
