# attempt at spiking neural network
# following https://arxiv.org/pdf/1907.13223.pdf

# A synaptic transfer function defines the dynamics of a neuron's membrane
# potential in response to a presynaptic spike. Equivalent to the activation
# function.
import Base.replace
using LambertW
using Flux
using Flux.Tracker: update!
using Flux: glorot_normal

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

function get_AI(ts, ws; τ::Float32=1f0)
    @assert length(ts) == length(ws)
    AI = 0
    for (ti, wi) in zip(ts, ws)
        AI += wi*exp(τ*ti)
    end
    return AI
end

function get_BI(ts, ws; τ::Float32=1f0)
    @assert length(ts) == length(ws)
    BI = 0
    for (ti, wi) in zip(ts, ws)
        BI += ti*wi*exp(τ*ti)
    end
    return BI
end

function t_out(thresh, ts, ws; τ::Float32=1f0)
    AI = get_AI(ts, ws)
    BI = get_BI(ts, ws)
    try # lambertw function throws error if argument < -e^-1 (neuron did not spike)
        t = BI/AI - (1/τ)*lambertw(-τ*(thresh/AI)*exp(τ*(BI/AI)))
        # t is only valid if it occurs later than the earliest input spike
        return (t>min(ts...)) ? t : NaN32
    catch ex
        return NaN32 # return NaN, meaning neuron did not spike
    end
end

function get_WI(thresh, ts, ws; τ::Float32=1f0)
    AI = get_AI(ts, ws)
    BI = get_BI(ts, ws)
    try
        return lambertw(-(thresh/AI)*exp(BI/AI))
    catch ex
        return NaN32
    end
end

function d_T_out_wrt_tj(thresh, ts, ws, tj, wj; τ::Float32=1f0)
    if isnan(tj) # if presynaptic neuron j did not spike, cannot compute the derivative
        return 0f0
    end

    fired::Array = []
    for (i, t) in enumerate(ts)
        if !isnan(t)  # if the value is not NaN
            push!(fired, i) # add the index to the list of neurons that fired
        end
    end

    ts = ts[fired] # only need the spike times of neurons that spiked
    ws = ws[fired] # only need those weights that correspond to those times

    WI = get_WI(thresh, ts, ws)
    AI = get_AI(ts, ws)
    BI = get_BI(ts, ws)
    (wj*exp(tj)*(tj-(BI/AI)+WI+1))/(AI*(1+WI))
end

function get_fired(a::Array)
    fired::Array = []
    for (i, val) in enumerate(a)
        if !isnan(val)  # if the value is not NaN
            push!(fired, i) # add the index to the list of neurons that fired
        end
    end
    return fired
end

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
    weights::Array # list of connected presynaptic neurons
    spike_time::Float32
    error::Float32
end

# constructs a neuron with randomly generated weights, no spike time, and no error
function Neuron(num_weights)
    ws = glorot_normal(num_weights)
    Neuron(ws, NaN32, 0f0)
end

struct Model
    layers::Array{Any,1}
    threshhold::Float32
    input_size::Int64
    τ::Float32
end

function Model(input_size::Int64, hidden_layer_sizes::Array{Int64,1}, thresh::Float32; τ::Float32=1f0)
    connections = vcat([input_size], hidden_layer_sizes[1:end-1]) # number of presynaptic inputs to each layer
    layers = [] # create the array to hold each layer
    # populate the each layer array with the specified number of neurons
    # each neuron needs a number of weights equal the number of presynaptic connections
    for (num_neurons, num_connections) in zip(hidden_layer_sizes, connections)
        push!(layers, [Neuron(num_connections) for i in 1:num_neurons])
    end
    Model(layers, thresh, input_size, τ)
end

###### Model functions ######

### Don't need AbstractArray types anymore...?
# takes a Model and an array of inputs, returns an array of final outputs, array of outputs by layer
function fwrd(m::Model, ts::AbstractArray)
    if length(ts) != m.input_size
        println("Size of input does not match the input layer!")
        return
    end
    input::AbstractArray = ts # starting input
    fired::Array = [i for i in 1:length(input)] # list of presynaptic neurons that fired (all for input)
    out::AbstractArray = [] # list of outputs from the layer
    layer_outputs::AbstractArray = []
        for (i, layer) in enumerate(m.layers) # for each layer in the model
        out = [] # clear the array
        for neuron in layer # for each neuron, calculate t_out and append to outputs
            push!(out, t_out(m.threshhold, input, neuron.weights[fired]))
        end
        push!(layer_outputs, out) # add this layers output to the list of outputs
        fired = [] # array to hold the indices of the neurons that fired in previous layer
        for ind in 1:length(out) # for the index of each value the layer output
            if !isnan(out[ind])  # if the value is not NaN
                push!(fired, ind) # add the index to the list of neurons that fired
            end
        end
        input = out[fired] # input to the next layer is output of this layer (only from neurons that fired)
    end
    return out, layer_outputs
end

function loss(m::Model, ts::Array, y::Array)
    out = fwrd(m, ts)[1]
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

# calculate the output then the derivative of error wrt output
function d_error_wrt_output(m::Model, ts::Array, y::Array)
    out = fwrd(m, ts)[1]
    out = replace(out, NaN32=>100f0) # replace NaN with large number so softmax and crossentropy will work
    out = [-val for val in out] # make output negative so that softmax returns greatest prob. for smallest val.
    cat_prob = softmax(out)
    ps = param(cat_prob)
    Flux.Tracker.back!(Flux.crossentropy(ps, y))
    Flux.Tracker.grad(ps)
end

# if output has already been calculated, get derivative of error wrt output
function d_error_wrt_output(out, y)
    out = replace(out, NaN32=>100f0) # replace NaN with large number so softmax and crossentropy will work
    out = [-val for val in out] # make output negative so that softmax returns greatest prob. for smallest val.
    cat_prob = softmax(out)
    ps = param(cat_prob)
    Flux.Tracker.back!(Flux.crossentropy(ps, y))
    Flux.Tracker.grad(ps)
end

# backpropagate the errors to update all weights
function backprop!(m::Model, ts::Array, y::Array)
    @assert m.input_size==length(ts) && length(m.layers[end])==length(y)
    out, layer_outputs = fwrd(m,ts)
    fired = get_fired(out) # get the indices of neurons that fired
    target_output_neuron = NaN # the output neuron whose spike time should be minimized
    for (i, val) in enumerate(y)
        if val == 1
            target_output_neuron = i
        end
    end
    @assert !isnan(target_output_neuron)
    ### println("neurons that fired: ", fired) # TEST
    ### println("derivative of error wrt output: ", d_error) # TEST
    # next, get the spike times of neurons in the layer before the output layer
    pst = layer_outputs[end-1] # pst (presynaptic spike times)
    grads::Array{Float32,1} = [] # will hold the derivative of the error wrt the weights

    ### OUTPUT LAYER
    for (i, neuron) in enumerate(m.layers[end]) # for the neurons in the output layer
        if i in fired # if the neuron fired
            mult = (i == target_output_neuron) ? 1 : -1
            println("mult = ", mult)
            grads = [] # clear the grads array
            for (j, weight) in enumerate(neuron.weights) # for each weight of the neuron
                push!(grads, mult*d_T_out_wrt_wj(m.threshhold, pst, neuron.weights, pst[j]))
            end
            println("grads for neuron ", i, " : ", grads, "\n")
            # update!(ADAM(), neuron.weights, grads)
        else # neuron did not fire
            # assign arbitrary positive value as the gradient, encouraging smaller weights
            # smaller weights push the input spikes earlier in time, increasing chance of spike
            grads = [100f0 for i in 1:length(m.layers[end-1])]
            update!(ADAM(), neuron.weights, grads)
        end
    end

    ### OTHER LAYERS
#    for layer in m.layers[1:end-1]
#        for (i, neuron) in enumerate(layer)

#        end
#    end
end

# duplicate for viewing
function d_T_out_wrt_wj(thresh, ts, ws, tj; τ::Float32=1f0)
    WI = get_WI(thresh, ts, ws)
    AI = get_AI(ts, ws)
    BI = get_BI(ts, ws)
    (exp(tj)*(tj-(BI/AI)+WI))/(AI*(1+WI))
end

m = Model(10, [15,5], .7f0)
ts = map(Float32, rand(10))
y = [1,0,0,0,0]

ts = map((t) -> t>.7f0 ? .1f0 : t, ts)

function get_m!(m, ts)
    while isempty(get_fired(fwrd(m, ts)[1]))
        m = Model(10, [15,5], .7f0)
    end
    return m
end

# special initialization of weights?

# running backprop
function test(m, ts, y)
    println("initial model output:")
    println(fwrd(m,ts)[1], "\n")
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
   println(fwrd(m,ts)[1], "\n")
   println("ending output layer weights of neuron 1:")
   println(m.layers[end][1].weights)
end
