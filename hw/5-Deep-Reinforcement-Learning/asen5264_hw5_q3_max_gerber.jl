using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
import POMDPs
using CommonRLInterface
using Flux
using CommonRLInterface.Wrappers: QuickWrapper

# Override to a discrete action space, and position and velocity observations rather than the matrix.
env = QuickWrapper(HW5.mc,
                   actions=[-1.0, -0.5, 0, 0.5, 1.0],
                   observe=mc->observe(mc)[1:2]
                  )

function dqn(env)

    # create your loss function for Q training here
    function loss(Q, s, a_ind, r, sp, done)
        if (done)
            return (r - Q(s)[a_ind])^2
        end
        gamma = 0.95
        return (r + gamma * maximum(frozen_Q(sp)) - Q(s)[a_ind])^2
        # make sure to take care of cases when the problem has terminated correctly
    end

    Q = Chain(Dense(2, 128, relu),
              Dense(128, length(actions(env))))
    frozen_Q = deepcopy(Q)

    opt = Flux.setup(ADAM(0.0005), Q)

    ϵ = 0.5

    function policy(s, counter)

        if counter % 10 == 0 && ϵ > 0.05
            ϵ = ϵ * 0.995
        end

        if rand() < ϵ
            return rand(1:length(actions(env)))
        else
            return argmax(Q(s))
        end

    end

    Q_hist = []
    buffer = []
    num_episodes = 500;
    R = []

    counter = 0

    max_steps = 400

    Q_best = Q;
    max_r = 0.0;

    for i = 1:num_episodes

        reset!(env)
        done = terminated(env)

        s = observe(env)

        num_steps = 0

        while !done && num_steps < max_steps

            if counter % 250 == 0
                frozen_Q = deepcopy(Q)
            end

            # We can create 1 tuple of experience like this
            a_ind = policy(s, counter) 
            r = act!(env, actions(env)[a_ind])
            sp = observe(env)
            done = terminated(env)

            push!(buffer, (s, a_ind, r, sp, done))
            buffer = length(buffer) > 100_000 ? buffer[2:end] : buffer

            if num_steps % 100 == 0 && num_steps > 0 && counter > 5_000
                for j = 1:100
                    data = rand(buffer, min(length(buffer), 20)) # changed from 10 to 20
                    Flux.Optimise.train!(loss, Q, data, opt)
                end
            end

            if (done)
                break
            end

            s = sp

            counter += 1
            num_steps += 1

        end

        r = 0.0

        for j in 1:100

            reset!(env)
            s = observe(env)

            done_j = terminated(env)

            if (done_j)
                break
            end

            num_steps_j = 0

            while !done_j && num_steps_j < max_steps

                a_ind = argmax(Q(s))
                r += act!(env, actions(env)[a_ind]) * 0.99^(num_steps-1)
                sp = observe(env)
                done_j = terminated(env)

                s = sp

                num_steps_j += 1

            end

        end

        push!(R,r/100)

        if r/100 > max_r && r/100 > 0 counter > 5_000
            max_r = r/100
            Q_best = Q

            buffer = buffer[end-num_steps:end]
        end

        println("episode: ", i, " term: ", done, " counter: ", counter, " max_r: ", max_r, " reward: ", r/100)

        if r/100 > 40
            HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], "mage7128@colorado.edu"; n_episodes=10_000, fname="results_"*string(i)*".json")
        end

        push!(Q_hist, deepcopy(Q))

    end
    
    return Q_hist, Q_best, R

end

Q_best, Q_best, R = dqn(env)