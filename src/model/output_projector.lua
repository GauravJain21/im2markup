require 'nn'

function createOutputUnit(input_size, output_size)
    local model = nn.Sequential()
    model:add(nn.Linear(input_size, output_size))
    model:add(nn.LogSoftMax())
    --[[This model takes input_size input, linearly 
		transforms it into output_size output using
		y = A * x + b, and finally applies log of 
		Softmax to this output
		]]
    return model
end
