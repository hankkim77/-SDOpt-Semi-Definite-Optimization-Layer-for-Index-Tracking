import torch
import torch.nn as nn
from functools import reduce
import operator

from utils import View


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())


class GlobalICLN(nn.Module):
    
    def __init__(
        self,
        x_dim,
        y_dim,
        u_dim,
        z_dim,
        output_dim=1,
        act_fn='SOFTPLUS',
        **kwargs
    ):
        super(GlobalICLN, self).__init__()
        
        if act_fn.upper()=='ELU':
            self.act_fn = nn.ELU()
        elif act_fn.upper()=='SOFTPLUS':
            self.act_fn = nn.Softplus()
        else:
            raise LookupError()
        
        # Input
        #   Upstream
        self.x_to_u = nn.Linear(x_dim, u_dim)
        #   Downstream
        self.x_to_ydim = nn.Linear(x_dim, y_dim)
        self.y_to_zdim = nn.Linear(y_dim, z_dim)
        self.x_to_zdim = nn.Linear(x_dim, z_dim)
        
        # Hidden (for later use)
        #   Upstream
        self.u_to_u = nn.Linear(u_dim, u_dim)
        #   Downstream
        #       1st Term
        self.u_to_zdim = nn.Linear(u_dim, z_dim)
        self.zdim_to_z = nn.Linear(z_dim, z_dim, bias=False)
        #       2nd Term
        self.u_to_ydim = nn.Linear(u_dim, y_dim)
        self.ydim_to_z = nn.Linear(y_dim, z_dim, bias=False)
        #       3rd Term
        self.u_to_z = nn.Linear(u_dim, z_dim)
        
        # Output
        #   Downstream
        #       1st Term
        self.out_u_to_zdim = nn.Linear(u_dim, z_dim)
        self.out_zdim_to_out = nn.Linear(z_dim, output_dim, bias=False)
        #       2nd Term
        self.out_u_to_ydim = nn.Linear(u_dim, y_dim)
        self.out_ydim_to_out = nn.Linear(y_dim, output_dim, bias=False)
        #       3rd Term
        self.out_u_to_out = nn.Linear(u_dim, output_dim)

        
    def forward(self, x, y):
        # Input
        #   Upstream
        u1 = self.x_to_u(x)
        u1 = self.act_fn(u1)
        #   Downstream
        xz1 = self.x_to_zdim(x)
        yz1 = self.y_to_zdim(y)
        z1 = self.act_fn(xz1 + yz1)
        
        # Hidden
        # no hid
        
        # Output
        #   Downstream
        #       1st Term
        uzdim = self.out_u_to_zdim(u1)
        uzdim = torch.clamp_min(uzdim,0) * z1
        uzz2 = self.out_zdim_to_out(uzdim)
        #       2nd Term
        uydim = self.out_u_to_ydim(u1)
        uydim *= y
        uyz2 = self.out_ydim_to_out(uydim)
        #       3rd Term
        uz2 = self.out_u_to_out(u1)
        
        out = self.act_fn(uzz2 + uyz2 + uz2)
        
        return out
        

class ICLN(nn.Module):
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=1,
        num_hidden_layers=1,
        act_fn='ELU',
    ):
        super(ICLN, self).__init__()
        
        if act_fn.upper()=='ELU':
            self.act_fn = nn.ELU()
        elif act_fn.upper()=='SOFTPLUS':
            self.act_fn = nn.Softplus()
        else:
            raise LookupError()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.act_fn,
        )
        
        self.hid_layers = nn.ModuleList()
        self.hid_layers_for_input = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hid_layers.append(
                PositiveLinear(hidden_dim, hidden_dim),
            )
            self.hid_layers_for_input.append(
                nn.Linear(input_dim, hidden_dim, bias=False),
            )
            
        self.output_layer = PositiveLinear(hidden_dim, output_dim)
        self.output_layer_for_input = nn.Linear(input_dim, output_dim, bias=False)

        
    def forward(self, y):
        z = self.input_layer(y)
        for hid_layer, hid_layer_for_input in zip(self.hid_layers, self.hid_layers_for_input):
            zh_1 = hid_layer(z)
            zh_2 = hid_layer_for_input(y)
            z = zh_1 + zh_2
            z = self.act_fn(z)
        zo_1 = self.output_layer(z)
        zo_2 = self.output_layer_for_input(y)
        z = zo_1 + zo_2
        z = self.act_fn(z)
        
        return z


class Quadratic(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(
        self,
        Y,  # true labels
        quadalpha=1e-3,  # regularisation weight
        **kwargs
    ):
        super(Quadratic, self).__init__()
        self.alpha = quadalpha
        self.Y_raw = Y.detach()
        self.Y = torch.nn.Parameter(self.Y_raw.view((-1)))
        self.num_dims = self.Y.shape[0]

        # Create quadratic matrices
        bases = torch.rand((self.num_dims, self.num_dims, 4)) / (self.num_dims * self.num_dims)
        self.bases = torch.nn.Parameter(bases)  

    def forward(self, Yhat):
        # Flatten inputs
        if len(Yhat.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[:-len(self.Y_raw.shape)], self.num_dims))

        # Measure distance between predicted and true distributions
        diff = (self.Y - Yhat).unsqueeze(-2)

        # Do (Y - Yhat)^T Matrix (Y - Yhat)
        basis = self._get_basis(Yhat).clamp(-10, 10)
        quad = (diff @ basis).square().sum(dim=-1).squeeze()

        return quad 

    def _get_basis(self, Yhats):
        # Figure out which entries to pick
        #   Are you above or below the true label
        direction = (Yhats > self.Y).type(torch.int64)
        #   Use this to figure out the corresponding index
        direction_col = direction.unsqueeze(-1)
        direction_row = direction.unsqueeze(-2)
        index = (direction_col + 2 * direction_row).unsqueeze(-1)

        # Pick corresponding entries
        bases = self.bases.expand(*Yhats.shape[:-1], *self.bases.shape)
        basis = bases.gather(-1, index).squeeze()
        return torch.tril(basis)


class DenseLoss(torch.nn.Module):
    """
    A Neural Network-based loss function
    """

    def __init__(
        self,
        Y,
        num_layers=4,
        hidden_dim=100,
        activation='relu'
    ):
        super(DenseLoss, self).__init__()
        # Save true labels
        self.Y = Y.detach().view((-1))
        # Initialise model
        self.model = dense_nn(Y.numel(), 1, num_layers, intermediate_size=hidden_dim, output_activation=activation)

    def forward(self, Yhats):
        # Flatten inputs
        Yhats = Yhats.view((-1, self.Y.numel()))

        return self.model(Yhats)


def dense_nn(
    num_features,
    num_targets,
    num_layers,
    intermediate_size=10,
    activation='relu',
    output_activation='sigmoid',
):
    if num_layers > 1:
        if intermediate_size is None:
            intermediate_size = max(num_features, num_targets)
        if activation == 'relu':
            activation_fn = torch.nn.ReLU
        elif activation == 'sigmoid':
            activation_fn = torch.nn.Sigmoid
        else:
            raise Exception('Invalid activation function: ' + str(activation))
        net_layers = [torch.nn.Linear(num_features, intermediate_size), activation_fn()]
        for _ in range(num_layers - 2):
            net_layers.append(torch.nn.Linear(intermediate_size, intermediate_size))
            net_layers.append(activation_fn())
        if not isinstance(num_targets, tuple):
            net_layers.append(torch.nn.Linear(intermediate_size, num_targets))
        else:
            net_layers.append(torch.nn.Linear(intermediate_size, reduce(operator.mul, num_targets, 1)))
            net_layers.append(View(num_targets))
    else:
        if not isinstance(num_targets, tuple):
            net_layers = [torch.nn.Linear(num_features, num_targets)]
        else:
            net_layers = [torch.nn.Linear(num_features, reduce(operator.mul, num_targets, 1)), View(num_targets)]

    if output_activation == 'relu':
        net_layers.append(torch.nn.ReLU())
    elif output_activation == 'sigmoid':
        net_layers.append(torch.nn.Sigmoid())
    elif output_activation == 'tanh':
        net_layers.append(torch.nn.Tanh())
    elif output_activation == 'softmax':
        net_layers.append(torch.nn.Softmax(dim=-1))

    return torch.nn.Sequential(*net_layers)

   
if __name__ == "__main__":

    pass
