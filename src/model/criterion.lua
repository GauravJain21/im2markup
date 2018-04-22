require 'nn'

function createCriterion(output_size)
    local weights = torch.ones(output_size)
    weights[1] = 0
    --for i = 4, 53+3 do
    --    weights[i] = 16
    --end
    criterion = nn.ClassNLLCriterion(weights)
    --[[NLL = negative log likelihood, weights are assigned
    	to classes because the training set is unbalanced.

    	Due to the behaviour of the backend code,
    	it is necessary to set sizeAverage to false 
    	when calculating losses in non-batch mode.
    	
    	Indeed, the ignoreIndex (defaults to -100) 
    	specifies a value for targets to be ignored. 
    	The commensurate gradInput for that target will 
    	be zero. When sizeAverage=true (the default), 
    	the gradInput and output are averaged over 
    	non-ignored targets.
    ]]
    criterion.sizeAverage = false

    return criterion
end
