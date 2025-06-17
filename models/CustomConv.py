
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
DIM=16


def float_to_fixed(value, integer_bits=16, fractional_bits=16):
    scale_factor = 2 ** fractional_bits
    return int(value * scale_factor)

def fixed_to_float(value, integer_bits=16, fractional_bits=16):
    scale_factor = 2 ** fractional_bits
    return value / scale_factor

def BitFlipFixedPoint(original_value, bit, integer_bits=16, fractional_bits=16):
    fixed_value = float_to_fixed(original_value, integer_bits, fractional_bits)
    bits = format(fixed_value, '032b')  # 32-bit representation
    new_bits = bits[:bit] + ('1' if bits[bit] == '0' else '0') + bits[bit + 1:]
    # Convert back to fixed-point integer
    new_fixed_value = int(new_bits, 2)

    # Convert back to floating-point
    new_value = fixed_to_float(new_fixed_value, integer_bits, fractional_bits)
    return new_value



def vectorized_float_to_fixed(float_array, integer_bits=16, fractional_bits=16):
    scale_factor = 2 ** fractional_bits
    return (float_array * scale_factor).numpy().astype(np.int32)

def vectorized_fixed_to_float(fixed_array, integer_bits=16, fractional_bits=16):
    scale_factor = 2 ** fractional_bits
    return fixed_array / scale_factor

def vectorized_bit_flip_fixed_point(float_array, bit_positions, integer_bits=12, fractional_bits=20):
    # Convert float array to fixed-point
    fixed_array = vectorized_float_to_fixed(float_array, integer_bits, fractional_bits)
    
    # Generate masks for bit flipping
    masks = 1 << bit_positions.numpy().astype(np.uint32)
    
    # Perform XOR operation to flip bits
    fixed_array ^= masks
    
    # Convert fixed-point array back to floating point
    flipped_floats = vectorized_fixed_to_float(fixed_array, integer_bits, fractional_bits)
    
    return torch.from_numpy(flipped_floats).float()

def vectorized_bit_flip(float_array, bit_positions):
    # Ensure float_array is in 32-bit floating point
    float_array = np.array(float_array, dtype=np.float32)
    # View the float array as 32-bit unsigned integers
    int_repr = float_array.view(np.uint32)
    
    # Ensure bit_positions is also in 32-bit unsigned integers and calculate the masks
    masks = np.left_shift(1, bit_positions.numpy().astype(np.uint32)).astype(np.uint32)

    # Perform XOR operation while both arrays are uint32
    int_repr ^= masks

    # View the modified integer array as floats again and return
    flipped_floats = int_repr.view(np.float32)
    return torch.from_numpy(flipped_floats)


InjectFaults = False

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.cin_in_tile = DIM // self.kernel_size
        self.cin_tiles = math.ceil(self.in_channels / self.cin_in_tile)
        self.cout_tiles = math.ceil(self.out_channels / DIM)
        print("|11||||||||||||||||||| ",self.cin_in_tile, self.cin_tiles, self.cout_tiles)
        
        # Initialize weights and biases for the convolutional filters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize the partial sums list with the expected dimensions
        self.ps = [[0 for _ in range(self.cin_tiles)] for __ in range(self.cout_tiles)]
        self.do_tile = False
        self.max_layer = 0
        self.layer_detect = 0
        self.tile_detect = 0
        self.total_fault = 0
        self.InjectFault = False
        
        # self.mini_conv = nn.Conv2d(256, 256, kernel_size=3, stride=self.stride, padding=0),
    def forward(self, x):
        batch_size, _, H, W = x.size()
        H_out = (H - self.kernel_size + 2 * self.padding) // self.stride + 1
        W_out = (W - self.kernel_size + 2 * self.padding) // self.stride + 1
        output = torch.zeros((batch_size, self.out_channels, H_out, W_out)).to(x.device)
        # print(H_out*W_out*self.out_channels)

        # transfer_weights(self, mytestconv)
        # my_output = mytestconv(x)
        
        # fi_ot = random.randint(0, self.cout_tiles-1)
        fi_it = torch.randint(0, self.cin_tiles, (self.out_channels,), device=output.device)
        for oo in range(self.cout_tiles):
            cout_lower = oo * DIM
            cout_upper = min((oo+1) * DIM,self.out_channels)
            for ii in range(self.cin_tiles):
                cin_lower = ii * self.cin_in_tile
                cin_upper = min((ii+1) * self.cin_in_tile,self.in_channels)
                self.weight[cout_lower:cout_upper,cin_lower:cin_upper,:,:]
                mytestconv = nn.Conv2d(cin_upper-cin_lower, cout_upper-cout_lower, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, bias=False)
                mytestconv.weight = nn.Parameter(self.weight[cout_lower:cout_upper,cin_lower:cin_upper,:,:].clone())
                o_temp = mytestconv(x[:,cin_lower:cin_upper,:,:])
                output[:,cout_lower:cout_upper,:,:] += o_temp
                # print("this is mm", mm)
                if(self.InjectFault == False):
                    mm = output[:,cout_lower:cout_upper,:,:].max().item()
                    self.ps[oo][ii] = max (self.ps[oo][ii], mm)
                    self.max_layer = max(self.max_layer, mm)
                else:
                    # print("NO WAY")
                    for oc in range(cout_lower,cout_upper):
                        if(fi_it[oc] == ii):
                            batch_size = output.shape[0]
                            depths = torch.randint(cout_lower, cout_upper, (batch_size,), device=output.device)
                            rows = torch.randint(0, output.shape[2], (batch_size,), device=output.device)
                            cols = torch.randint(0, output.shape[3], (batch_size,), device=output.device)
                            bits = torch.randint(0, 32, (batch_size,), device=output.device)
                            batch_indices = torch.arange(batch_size, device=output.device)
                            newVs = vectorized_bit_flip_fixed_point( output[batch_indices, depths, rows, cols],bits) 
                            output[batch_indices, depths, rows, cols] = newVs
                            self.layer_detect += torch.sum(newVs>self.max_layer).item()
                            self.tile_detect += torch.sum(newVs>self.ps[oo][ii]).item()
                            self.total_fault += output.shape[0] 



                    if(self.do_tile):
                        isNormal = output[:,cout_lower:cout_upper,:,:] <= self.ps[oo][ii] #self.max_layer# 
                        output[:,cout_lower:cout_upper,:,:] *= isNormal
                    else:
                        isNormal = output[:,cout_lower:cout_upper,:,:] <= self.max_layer# self.ps[oo][ii] #
                        output[:,cout_lower:cout_upper,:,:] *= isNormal
                    


        for o in range(self.out_channels):
            output[:,o,:,:] += self.bias[o] 
        return output


def transfer_weights(original_conv_layer, custom_conv_layer):
    with torch.no_grad():  # We don't want these operations to be recorded for further gradient computations
        custom_conv_layer.weight = nn.Parameter(original_conv_layer.weight.clone())
        if original_conv_layer.bias is not None:
            custom_conv_layer.bias = nn.Parameter(original_conv_layer.bias.clone())
layer_cnt = 0
def replace_conv_layers(model):
    global layer_cnt
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            layer_cnt += 1
            # if layer_cnt!=2:
            #     continue
            # print("name=",name)
            # Match the parameters of the original layer
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]  # Assuming square kernels
            stride = module.stride[0]  # Assuming square stride
            padding = module.padding[0]  # Assuming square padding
            
            # Create a new instance of CustomConv2d with the same parameters
            custom_conv = CustomConv2d(in_channels, out_channels, kernel_size, stride, padding)
            
            # Transfer the weights from the original convolutional layer
            transfer_weights(module, custom_conv)
            
            # Replace the layer in the model
            setattr(model, name, custom_conv)
            
        else:
            # Recursively apply this function to child modules
            replace_conv_layers(module)
    return model
