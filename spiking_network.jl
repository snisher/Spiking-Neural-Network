# attempt at spiking neural network
# following https://arxiv.org/pdf/1907.13223.pdf

# A synaptic transfer function defines the dynamics of a neuron's membrane
# potential in response to a presynaptic spike. Equivalent to the activation
# function.
import Base.replace
using LambertW
using Flux
using Flux.Tracker: update!
using Flux: glorot_normal, onecold
using Distributions: Uniform

######## helper functions ########

# takes the current time, a set of arrival times of post synaptic inputs (ts),
# and the weights of the connections (ws)
function v_mem(t, ts, ws; τ::Float32=1f0)
    # calculates the membrane potential at time t given the inputs and weights
    @assert length(ts) == length(ws)
    order = sortperm(ts) # get the order of inputs in time
    ts = ts[order] # sort by spike time
    ws = ws[order] # sort to match order of spike times
    v = 0
    for (ti, wi) in zip(ts, ws)
        if ti > t # only need to calculate up to the current time. Assumes times are in order
            return v
        end
        v += wi*(t-ti)*exp(τ*(ti-t))
    end
    return v
end

# AI is used in computation of t_out
function get_AI(ts, ws; τ::Float32=1f0)
    @assert length(ts) == length(ws)
    AI = 0
    for (ti, wi) in zip(ts, ws)
        AI += wi*exp(τ*ti)
    end
    return AI
end

# BI is used in computation of t_out
function get_BI(ts, ws; τ::Float32=1f0)
    @assert length(ts) == length(ws)
    BI = 0
    for (ti, wi) in zip(ts, ws)
        BI += ti*wi*exp(τ*ti)
    end
    return BI
end

# calculates the time at which a neuron spiked given its threshhold and inputs
# returns NaN if it did not spike
function t_out(thresh, ts, ws; τ::Float32=1f0)
    AI = get_AI(ts, ws)
    BI = get_BI(ts, ws)
    # The derivative of the activation function at the time of a spike is
    # equal to AI * (never negative number). Thus, if AI is negative, then
    # any possible spike time to be calculated will be at a time when the
    # membrane potential is decreasing, and is not a valid spike.
    if (AI < 0) # not valid if AI is negative
        return NaN32
    end
    try # lambertw function throws error if argument < -e^-1 (neuron did not spike)
        t = BI/AI - (1/τ)*lambertw(-τ*(thresh/AI)*exp(τ*(BI/AI)))
        # t is only valid if it occurs later than the earliest input spike
        return (t>min(ts...)) ? t : NaN32
    catch ex
        return NaN32 # return NaN, meaning neuron did not spike
    end
end

# WI is used in computation of the derivative of t_out
function get_WI(thresh, ts, ws; τ::Float32=1f0)
    AI = get_AI(ts, ws)
    BI = get_BI(ts, ws)
    try
        return lambertw(-(thresh/AI)*exp(BI/AI))
    catch ex
        return NaN32
    end
end

# returns the indices of the neurons that fired,
# ie the indices of the array which are not NaN
function get_fired(a::Array)
    fired = Int64[]
    for (i, val) in enumerate(a)
        if !isnan(val)  # if the value is not NaN
            push!(fired, i) # add the index to the list of neurons that fired
        end
    end
    return fired
end

# derivative of t_out wrt a certain time
# used for adjusting the bias
function d_T_out_wrt_tj(thresh, ts, ws, tj, wj; τ::Float32=1f0)
    if isnan(tj) # if presynaptic neuron j did not spike, cannot compute the derivative
        return 0f0
    end

    fired = get_fired(ts)

    ts = ts[fired] # only need the spike times of neurons that spiked
    ws = ws[fired] # only need those weights that correspond to those times

    WI = get_WI(thresh, ts, ws)
    AI = get_AI(ts, ws)
    BI = get_BI(ts, ws)

    (wj*exp(tj)*(tj-(BI/AI)+WI+1))/(AI*(1+WI))
end

# derivative of a postsynaptic spike time wrt to a presynaptic spike time and corresponding weight
function d_T_out_wrt_wj(thresh, ts, ws, tj; τ::Float32=1f0)
    if isnan(tj) # if presynaptic neuron j did not spike, cannot compute the derivative
        return 0f0
    end

    fired = get_fired(ts)

    ts = ts[fired] # only need the spike times of neurons that spiked
    ws = ws[fired] # only need those weights that correspond to those times

    WI = get_WI(thresh, ts, ws)
    AI = get_AI(ts, ws)
    BI = get_BI(ts, ws)
    (exp(tj)*(tj-(BI/AI)+WI))/(AI*(1+WI))
end

########## model #############

mutable struct Neuron
    weights::Array{Float32} # list of weights from connected presynaptic neurons
    bias::Float32
end

# constructs a neuron with randomly generated weights
function Neuron(num_weights)
    ws = glorot_normal(num_weights)
    bias = 0f0 # rand(Uniform(0f0, 1f0), 1) # bias not used as of now
    Neuron(ws, bias)
end

struct Model{T<:Array{Array{Neuron,1},1}}
    layers::T
    threshhold::Float32
    input_size::Int64
    τ::Float32
end

# constructs a model with layers of neurons
function Model(input_size::Int64, hidden_layer_sizes::Array{Int64,1}, thresh::Float32; τ::Float32=1f0)
    connections = vcat([input_size], hidden_layer_sizes[1:end-1]) # number of presynaptic inputs to each layer
    layers = Array{Neuron,1}[] # create the array to hold each layer (array of arrays of neurons)
    # populate each layer array with the specified number of neurons
    # each neuron needs a number of weights equal the number of presynaptic connections
    for (num_neurons, num_connections) in zip(hidden_layer_sizes, connections)
        push!(layers, [Neuron(num_connections) for i in 1:num_neurons])
    end
    Model(layers, thresh, input_size, τ)
end

###### Model functions ######

### Don't need AbstractArray types anymore...?
# takes a Model and an array of inputs, returns an array of final outputs, and array of outputs by layer for debug
function fwd(m::Model, ts::AbstractArray)
    if length(ts) != m.input_size
        println("Size of input does not match the input layer!")
        return
    end
    input::AbstractArray = ts # starting input
    fired = [i for i in 1:length(input)] # list of presynaptic neurons that fired (all for input)
    out::AbstractArray = [] # list of outputs from the layer
    layer_outputs::AbstractArray = []
    for (i, layer) in enumerate(m.layers) # for each layer in the model
        out = [] # clear the array
        for neuron in layer # for each neuron, calculate t_out and append to outputs
            t_with_bias = t_out(m.threshhold, input, neuron.weights[fired]) + neuron.bias
            push!(out, t_with_bias)
        end
        push!(layer_outputs, out) # add this layer's output to the list of outputs
        fired = get_fired(out) # array to hold the indices of the neurons that fired in previous layer
        input = out[fired] # input to the next layer is output of this layer (only from neurons that fired)
    end
    return out, layer_outputs
end

function loss(m::Model, ts::Array, y::Array)
    out = fwd(m, ts)[1]
    out = replace(out, NaN32=>100f0) # replace NaN with large number so softmax and crossentropy will work
    cat_prob = softmax([-val for val in out])
    Flux.crossentropy(cat_prob, y)
end

# calculate loss if output is already calculated
function loss(out::AbstractArray, y::Array)
    out = replace(out, NaN32=>100f0) # replace NaN with large number so softmax and crossentropy will work
    cat_prob = softmax([-val for val in out])
    Flux.crossentropy(cat_prob, y)
end

# calculate the output, then the derivative of error wrt output
# gradient is positive for target output neuron and negative for others
function d_error_wrt_output(m::Model, ts::Array, y::Array)
    out = fwd(m, ts)[1]
    out = replace(out, NaN32=>100f0) # replace NaN with large number so softmax and crossentropy will work
    out = [-val for val in out] # make output negative so that softmax returns greatest prob. for smallest val.
    cat_prob = softmax(out)
    grad = cat_prob - y
    grad = -grad/length(y) # negative of grad bc input to softmax was negative
end

# derivative if output has already been calculated
# gradient is positive for target output neuron and negative for others
function d_error_wrt_output(out, y)
    out = replace(out, NaN32=>100f0) # replace NaNs with large number so softmax and crossentropy will work
    out = [-val for val in out] # make output negative so that softmax returns greatest prob. for smallest val.
    cat_prob = softmax(out)
    grad = cat_prob - y
    grad = -grad/length(y) # negative of grad bc input to softmax was negative
end

# backpropagate the errors to update all weights
# goal: make the target neuron fire in shorter time.
# Make non-target neurons fire in longer time.
# make neurons that are not currently firing fire.
function backprop!(m::Model, ts::Array, y::Array)
    @assert m.input_size==length(ts) && length(m.layers[end])==length(y)
    out, layer_outputs = fwd(m,ts)
    fired = get_fired(out) # get the indices of neurons that fired
    target_output_neuron = onecold(y) # the output neuron whose spike time should be minimized
    @assert !isnan(target_output_neuron)

    ## println("neurons that fired: ", fired) # TEST
    ### println("derivative of error wrt output: ", d_error) # TEST

    # next, get the spike times of neurons in the layer before the output layer
    pst = layer_outputs[end-1] # pst (presynaptic spike times)
    grads = Float32[] # will hold the derivative of the error wrt the weights

    ### OUTPUT LAYER
    dlwo = d_error_wrt_output(out, y) # derivative of loss wrt to each output
    for (i, neuron) in enumerate(m.layers[end]) # for each neuron in the output layer
        if i in fired # if the neuron fired
            grads = [] # clear the grads array
            for (j, weight) in enumerate(neuron.weights) # for each connection to that neuron
                push!(grads, dlwo[i]*d_T_out_wrt_wj(m.threshhold, pst, neuron.weights, pst[j]))
            end
            update!(ADAM(), neuron.weights, grads)
        else # neuron did not fire
            # Negative gradients so that weights become bigger.
            # Bigger weights = larger inputs and more likely to spike
            grads = [-100f0 for weight in neuron.weights]
            update!(ADAM(), neuron.weights, grads)
        end
    end

"""
plan for hidden layers: store error of each neurons output bc they
need to be used in the calculation. gradient should be:
d_T_out_wrt_wj() * error
error needs to be calculated so that it reflects all the "wants" of the neurons
in the next layer...
"""

    ### HIDDEN LAYERS
    for layer in m.layers[1:end-1]
        for (i, neuron) in enumerate(layer)
            if i in fired # if the neuron fired
                grads = []
                for (j, weight) in enumerate(neuron.weights)
                    #### stuff
                end
                # update weights
            else # neuron did not fire
                grads = [] # clear array
                grads = [-100f0 for w in neuron.weights]
                update!(ADAM(), neuron.weights, grads) # update the weight
            end
        end
    end
end

# duplicate for viewing
function d_T_out_wrt_wj(thresh, ts, ws, tj; τ::Float32=1f0)
    WI = get_WI(thresh, ts, ws)
    AI = get_AI(ts, ws)
    BI = get_BI(ts, ws)
    (exp(tj)*(tj-(BI/AI)+WI))/(AI*(1+WI))


ts = map(Float32, rand(10)) # random input times
y = Int64[1,0,0,0,0] # labels

ts = map((t) -> t>.7f0 ? .1f0 : t, ts) # make large ts smaller

# change model params until a model gets at least one non-NaN output
function get_m(ts)
    m = Model(10, [15,5], .7f0) # create model with 10 inputs, 1 hidden layer, threshhold of .7
    while isempty(get_fired(fwd(m, ts)[1]))
        m = Model(10, [15,5], .7f0)
    end
    return m
end

# special initialization of weights?

# running backprop
function test(m, ts, y)
    println("initial model output:")
    println(fwd(m,ts)[1], "\n")
    println("initial output layer weights of neuron 1:")
    println(m.layers[end][1].weights, "\n")
    i = 0
    l = loss(m,ts,y)
    new_l = loss(m,ts,y)
    while new_l == l
        i += 1
        if i > 100
            println("too many iterations, exiting\n")
            break
        end
        backprop!(m,ts,y) # run backprop
        new_l = loss(m,ts,y) # get the new loss
        if new_l != l
            println("loss changed after ", i, " backpropogations.")
            println("loss: ", new_l, "\n")
            break
        end
   end
   println("ending model output:")
   println(fwd(m,ts)[1], "\n")
   println("ending output layer weights of neuron 1:")
   println(m.layers[end][1].weights)
end


# Interestingly, the t_out approximation given in the paper occasionally results
# in a positive output time even when all the weights of the neuron are
# negative and the neuron membrane potential never exceeds zero... From looking
# at the paper's code, I modified t_out so that if the AI argument
# is negative then there is no spike which should fix the this problem.

"""
11/24/19
Updating weights of output layer neurons works, except that non-target output
neurons are encouraged to have smaller weights, which means after many backprops
they end up fluctuating: at one fwd() they fire, but after backprop!() they no longer
fire due to smaller weights, and after the next backprop!() they fire again after
the weights are increased to encourage firing. Maybe test whether the neuron is
close to not firing by checking its v_mem??

I am now trying to get backprop!() working for the other hidden layers.
"""

# derivative of cross entropy and softmax,
# from https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax
def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    m = y.shape[0]
    grad = softmax(X) # get the softmax of the output
    grad = grad - y # subtract the target from the softmax'd output
    grad = grad/m # divide by the number of classes
    return grad
