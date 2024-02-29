using DMUStudent.HW2
using POMDPs: states, stateindex, actions, discount
using POMDPTools: ordered_states
using LinearAlgebra

##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW2 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

=#

############
# Question 3
############

# @show actions(grid_world) # prints the actions. In this case each action is a Symbol. Use ?Symbol to find out more.

# T = transition_matrices(grid_world)
# display(T) # this is a Dict that contains a transition matrix for each action

# @show T[:left][1, 2] # the probability of transitioning between states with indices 1 and 2 when taking action :left

# R = reward_vectors(grid_world)
# display(R) # this is a Dict that contains a reward vector for each action

# @show R[:right][1] # the reward for taking action :right in the state with index 1

function value_iteration(m)
    
    A = collect(actions(m))
    R = reward_vectors(m)
    T = transition_matrices(m; sparse=true)
    γ = discount(m)
    
    V       = rand(length(states(m)))
    V_prime = rand(length(states(m)))
    
    epsilon = 1e-10
    
    V_temp  = zeros(length(states(m)), length(A))
    
    while (norm(V - V_prime, Inf) > epsilon)
        
        V .= V_prime 
        
        for (j, a) in enumerate(A)
            V_temp[:,j] = R[a] + γ * T[a] * V
        end
        V_prime[:] = maximum(V_temp, dims=2)
    end
    return V_prime
end

V = value_iteration(grid_world)
display(render(grid_world, color=V))

# You can use the following commented code to display the value. If you are in an environment with multimedia capability (e.g. Jupyter, Pluto, VSCode, Juno), you can display the environment with the following commented code. From the REPL, you can use the ElectronDisplay package.
# display(render(grid_world, color=V))

############
# Question 4
############

# You can create an mdp object representing the problem with the following:
n = 2
m = UnresponsiveACASMDP(n)

V = value_iteration(m)

@show HW2.evaluate(V, "mage7128@colorado.edu")