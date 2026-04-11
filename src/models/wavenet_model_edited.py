import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNetModel(nn.Module):
    def __init__(self, 
                 num_features,      # CHANGED: Number of variables in your WTP dataset
                 layers,          # CHANGED: 3 layers: receptive field = 8 days (dilations of 1,2,4). 5 layers: receptive field = 32 days (dilations of 1,2,4,8,16). Adjust based on how far back you want the model to look. 
                 blocks,          # CHANGED: Reduced from 4 to prevent overfitting on small windows
                 dilation_channels, #calculate 32 types of features for every day in window
                 residual_channels, #if training loss doesn't decrease, try increasing this to 64 or 128
                 skip_channels, #CHANGED: Reduced from 128 to prevent overfitting
                 end_channels, #CHANGED: Reduced from 128 to prevent overfitting
                 kernel_size,
                 dropout_rate = 0.2, #ADDED
                 bias=False):
        
        super(WaveNetModel, self).__init__()
        self.layers = layers
        self.blocks = blocks
        self.kernel_size = kernel_size
        
        # CHANGED: Receptive field calculation
        self.receptive_field = blocks * (2**layers - 1) * (kernel_size - 1) + 1
        
        self.dilations = []
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # CHANGED: Input is now continuous features, not 256 classes
        #1x1 convolution that projects raw input features into higher dimension (residual_channels)
        self.start_conv = nn.Conv1d(in_channels=num_features, 
                                    out_channels=residual_channels, 
                                    kernel_size=1, 
                                    bias=bias)

        for b in range(blocks):
            new_dilation = 1
            for i in range(layers):
                self.dilations.append(new_dilation)
                
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=kernel_size, dilation=new_dilation, bias=bias))
                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=kernel_size, dilation=new_dilation, bias=bias))
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=1, bias=bias))
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=1, bias=bias))
                
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels, out_channels=end_channels, kernel_size=1, bias=True)
        # CHANGED: Output is 1 channel (the predicted Turbidity value)
        self.end_conv_2 = nn.Conv1d(in_channels=end_channels, out_channels=1, kernel_size=1, bias=True)

    def forward(self, x):
        # x expected shape: (Batch_Size, Num_Features, Window_Size)
        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            residual = x
            
            # Causal padding (pad the left side of the sequence)
            pad_size = (self.kernel_size - 1) * self.dilations[i]
            x_padded = F.pad(x, (pad_size, 0))
            
            filter = torch.tanh(self.filter_convs[i](x_padded))
            gate = torch.sigmoid(self.gate_convs[i](x_padded))
            x = filter * gate


            x = self.dropout(x) # ADDED: Dropout for regularization

            s = self.skip_convs[i](x)
            skip = skip + s

            x = self.residual_convs[i](x)
            x = x + residual

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x) # Shape: (Batch_Size, 1, Window_Size)
        
        # CHANGED: We only want the prediction for the *last* time step of the window
        return x[:, :, -1].squeeze() # Shape: (Batch_Size,)