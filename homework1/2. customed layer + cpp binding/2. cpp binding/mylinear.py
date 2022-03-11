import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

import mylinear_cpp

class myLinearFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        # output = input.mm(weight.t())
        output, = mylinear_cpp.forward(input, weight)
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        # grad_input = grad_weight = None
        # grad_input = grad_output.mm(weight)
        # grad_weight = grad_output.t().mm(input)
        grad_input, grad_weight = mylinear_cpp.backward(grad_output, input, weight)

        mylinear_cpp.cppprint(input) # my implementation

        return grad_input, grad_weight


class myLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(myLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, input):
        return myLinearFunction.apply(input, self.weight)