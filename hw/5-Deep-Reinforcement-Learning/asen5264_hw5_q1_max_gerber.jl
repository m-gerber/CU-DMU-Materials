using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
import POMDPs

cancer = QuickPOMDP(
    states = [:healthy, :in_situ_cancer, :invasive_cancer, :death],
    actions = [:wait, :test, :treat],
    observations = [:positive, :negative],

    # transition should be a function that takes in s and a and returns the distribution of s'
    transition = function (s, a)
        if s == :healthy
            return SparseCat([:healthy, :in_situ_cancer], [0.98, 0.02])
        elseif s == :in_situ_cancer
            if a == :treat
                return SparseCat([:healthy, :in_situ_cancer], [0.6, 0.4])
            else
                return SparseCat([:in_situ_cancer, :invasive_cancer], [0.9, 0.1])
            end
        elseif s == :invasive_cancer
            if a == :treat
                return SparseCat([:healthy, :invasive_cancer, :death], [0.2, 0.6, 0.2])
            else
                return SparseCat([:invasive_cancer, :death], [0.4, 0.6])
            end
        end
        return Deterministic(s)
    end,

    # observation should be a function that takes in s, a, and sp, and returns the distribution of o
    observation = function (s, a, sp)
        if a == :test
            if sp == :healthy
                return SparseCat([:positive, :negative], [0.05, 0.95])
            elseif sp == :in_situ_cancer
                return SparseCat([:positive, :negative], [0.8, 0.2])
            elseif sp == :invasive_cancer
                return Uniform([:positive])
            end
        elseif a == :treat
            if s == :in_situ_cancer || s == :invasive_cancer
                return Uniform([:positive])
            end
        end
        return Uniform([:negative])
    end,

    reward = function (s, a)
        if s == :death
            return 0.0
        elseif a == :wait
            return 1.0
        elseif a == :test
            return 0.8
        elseif a == :treat
            return 0.1
        end
    end,

    initialstate = Uniform([:healthy]),

    discount = 0.99
)

@show

# evaluate with a random policy
policy = FunctionPolicy(o->POMDPs.actions(cancer)[1])
sim = RolloutSimulator(max_steps=100)
@show @time mean(POMDPs.simulate(sim, cancer, policy) for _ in 1:10_000)