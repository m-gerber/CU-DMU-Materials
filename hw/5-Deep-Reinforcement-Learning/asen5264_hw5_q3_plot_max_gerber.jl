using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
import POMDPs
import POMDPTools
using CommonRLInterface
using Flux
using Plots
using CommonRLInterface.Wrappers: QuickWrapper

function compute_reward(Q_hist)

    R = []

    for i in eachindex(Q_hist)

        r = 0.0

        for j in 1:100

            reset!(env)
            s = observe(env)

            done = terminated(env)

            if (done)
                break
            end

            num_steps = 0

            while !done && num_steps < 400

                a_ind = argmax(Q_hist[i](s))
                r += act!(env, actions(env)[a_ind]) * 0.99^(num_steps-1)
                sp = observe(env)
                done = terminated(env)

                s = sp

                num_steps += 1

            end

        end

        push!(R, r/100)

        println("episode: ", i, " discounted reward: ", r/100)

    end

    return R

end

# R = compute_reward(Q_hist)

Q = Q_best 

# HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], "mage7128@colorado.edu"; n_episodes=10_000) # you will need to remove the n_episodes=100 keyword argument to create a json file; evaluate needs to run 10_000 episodes to produce a json

#----------
# Rendering
#----------

plot1 = plot(xlabel="episode", ylabel="avg return")
plot!(plot1, 1:length(R), R, label="reward")
display(plot1)

# You can show an image of the environment like this (use ElectronDisplay if running from REPL):
display(render(env))

# The following code allows you to render the value function
xs = -3.0f0:0.1f0:3.0f0
vs = -0.3f0:0.01f0:0.3f0
heatmap(xs, vs, (x, v) -> maximum(Q([x, v])), xlabel="Position (x)", ylabel="Velocity (v)", title="Max Q Value")