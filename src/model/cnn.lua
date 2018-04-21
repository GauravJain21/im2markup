function createCNNModel(use_cuda)
    --use_cuda is a flag used to determine whether the training is being performed on a GPU or not
    local model = nn.Sequential()
    --sequential model, analogous to keras sequential

    -- input shape: (batch_size, 1, imgH, imgW)
    -- CNN part
    model:add(nn.AddConstant(-128.0))
    model:add(nn.MulConstant(1.0 / 128))
    --[[
        batch_size = 1
        kernel size = 3 x 3
        stride = 1
        padding = 1
    ]]
    --function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    --[[
    SpatialConvolution
    module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])
    Applies a 2D convolution over an input image composed of several input planes. The input tensor in forward(input) is expected to be a 3D tensor (nInputPlane x height x width).

    The parameters are the following:

    nInputPlane: The number of expected input planes in the image given into forward().
    nOutputPlane: The number of output planes the convolution layer will produce.
    kW: The kernel width of the convolution
    kH: The kernel height of the convolution
    dW: The step of the convolution in the width dimension. Default is 1.
    dH: The step of the convolution in the height dimension. Default is 1.
    padW: Additional zeros added to the input plane data on both sides of width axis. Default is 0. (kW-1)/2 is often used here.
    padH: Additional zeros added to the input plane data on both sides of height axis. Default is 0. (kH-1)/2 is often used here.
    ]]
    model:add(cudnn.SpatialConvolution(1, 64, 3, 3, 1, 1, 1, 1)) -- (batch_size, 64, imgH, imgW)
    --3 x 3 kernel with stride = 1, and borders padded with zeros to preserve the original input size
    model:add(cudnn.ReLU(true))
    --[[
        function SpatialMaxPooling:__init(kW, kH, dW, dH, padW, padH)
            self.padW = padW or 0
            self.padH = padH or 0
        end
    ]]
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- (batch_size, 64, imgH/2, imgW/2), here padding is 0 by default
    --[[
        2 x 2 --> 1 x 1
         _ _        _
        |_|_| ---> |_|
        |_|_|

    ]]

    --kW = kH = dW = dH = 2
    --64 feature maps are input and 128 are outputted
    model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)) -- (batch_size, 128, imgH/2, imgW/2)
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- (batch_size, 128, imgH/2/2, imgW/2/2)

    model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)) -- (batch_size, 256, imgH/2/2, imgW/2/2)
    --Batch normalization used
    model:add(nn.SpatialBatchNormalization(256))
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)) -- (batch_size, 256, imgH/2/2, imgW/2/2)
    model:add(cudnn.ReLU(true))
    
    model:add(cudnn.SpatialMaxPooling(1, 2, 1, 2, 0, 0)) -- (batch_size, 256, imgH/2/2/2, imgW/2/2)
    --[[
        Only the height gets reduced by half, no effect on the width of the input
        1 x 2 ---> 1 x 1
         _          _
        |_|   ---> |_|
        |_|

    ]]


    model:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)) -- (batch_size, 512, imgH/2/2/2, imgW/2/2)
    model:add(nn.SpatialBatchNormalization(512))
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialMaxPooling(2, 1, 2, 1, 0, 0)) -- (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
    --[[
        Only the width gets reduced by half, no effect on the height of the input
        2 x 1 ---> 1 x 1
         _ _        _
        |_|_| ---> |_|

    ]]
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
    model:add(nn.SpatialBatchNormalization(512))
    model:add(cudnn.ReLU(true))

    -- (batch_size, 512, H, W)    

    model:add(nn.Transpose({2, 3}, {3,4})) -- (batch_size, H, W, 512)
    --[[
        module = nn.Transpose({dim1, dim2} [, {dim3, dim4}, ...])
        Swaps dimension dim1 with dim2, then dim3 with dim4, and so on.

        Order of transpose:
        1. swap 512 with H (2 <-> 3)
            Now, H is at position 2 and 512 at position 3
        2. swap 512 with W (3 <-> 4)
    ]]
    model:add(nn.SplitTable(1, 3)) -- #H list of (batch_size, W, 512)
    --[[
        Not very clear to me for now.
        module = SplitTable(dimension, nInputDims)
Creates a module that takes a Tensor as input and outputs several tables, splitting the Tensor along the specified dimension. The optional parameter nInputDims allows to specify the number of dimensions that this module will receive. This makes it possible to forward both minibatch and non-minibatch Tensors through the same module.
    ]]

    --model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', false, true))
    -- if training on GPU, uncomment these lines
    --model:cuda()
    return model

end
