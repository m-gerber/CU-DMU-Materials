using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate, stateindex
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime

q_2 = true
q_3 = true
q_4 = true

############
# Question 2
############

function rollout(mdp, policy_function, s0, max_steps=100)
    r_total = 0.0
    t = 0
    while !isterminal(mdp,s0) && t < max_steps
        a = policy_function(mdp,s0)
        s0, r = @gen(:sp, :r)(mdp,s0,a)
        r_total += discount(mdp)^(t-1) * r
        t += 1
    end

    return r_total
end

function heuristic_policy(m, s)
    return rand(actions(m))
end

function my_policy(m, s)
    x = s[1]
    y = s[2]

    if x > 40 || (x > 20 && x < 31)
        return :left
    elseif x < 20 || (x > 30 && x < 40)
        return :right
    elseif y > 40 || (y > 20 && y < 31)
        return :down
    else
        return :up
    end
end

if q_2
    m_2 = HW3.DenseGridWorld(seed=3)

    n_runs_2 = 500

    initial_results = [rollout(m_2, heuristic_policy, rand(initialstate(m_2))) for _ in 1:n_runs_2]

    println()
    println("            ### QUESTION 2: ###")
    println("initial results:     | ",mean(initial_results))
    println("initial results sem: |   ",std(initial_results)/sqrt(length(initial_results)))

    my_results = [rollout(m_2, my_policy, rand(initialstate(m_2))) for _ in 1:n_runs_2]

    println("my results:          | ",mean(my_results))
    println("my results sem:      |   ",std(initial_results)/sqrt(length(initial_results)))
    println("---------------------|---------------------")
    println("improvement:         |  ",mean(my_results)-mean(initial_results))
    println()
end

############
# Question 3
############

bonus(nsa, ns) = nsa == 0 ? Inf : sqrt(log(ns)/nsa)

function simulate!(m, n, q, s, d, t=nothing)

    if d == 0 || isterminal(m,s)
        return rollout(m, my_policy, s)
    end

    if !haskey(n, (s, first(actions(m))))
        for a in actions(m)
            n[(s, a)] = 0
            q[(s, a)] = 0.0
        end
        return rollout(m, my_policy, s)
    end

    ns = sum(n[(s,a)] for a in actions(m))

    c = 200.0
    a = argmax(a->q[(s,a)] + c*bonus(n[(s,a)], ns), actions(m))

    sp, r = @gen(:sp, :r)(m,s,a)

    q_val = r + discount(m)*simulate!(m, n, q, sp, d-1, t)

    n[(s, a)] += 1
    q[(s, a)] += (q_val-q[(s, a)])/n[(s, a)]
    t[(s, a, sp)] = get(t, (s, a, sp), 0) + 1
    
    return q_val

end

if q_3
    m_3 = DenseGridWorld(seed=3)
    
    n_sims_3 = 7

    d_3 = 50

    S = statetype(m_3)
    A = actiontype(m_3)

    n_3 = Dict{Tuple{S, A}, Int}()
    q_3 = Dict{Tuple{S, A}, Float64}()
    t_3 = Dict{Tuple{S, A, S}, Int}()

    s_3 = SA[19,19]
    for _ in 1:n_sims_3
        simulate!(m_3, n_3, q_3, s_3, d_3, t_3)
    end

    inchrome(visualize_tree(q_3, n_3, t_3, s_3))
end

############
# Question 4
############

function select_action(m, s)

    start = time_ns()

    S = statetype(m)
    A = actiontype(m)

    n = Dict{Tuple{S, A}, Int}()
    q = Dict{Tuple{S, A}, Float64}()
    t = Dict{Tuple{S, A, S}, Int}()

    d = 40
    n_sims = 2000

    i = 0

    while time_ns() < start + 40_000_000 && i < n_sims
        simulate!(m, n, q, s, d, t)
        i += 1
    end

    return argmax(a->q[(s, a)], actions(m))    

end

if q_4
    m_4 = DenseGridWorld(seed=4)

    n_MCTS_4 = 100
    n_steps_4 = 100

    reward_4 = zeros(Float64, n_MCTS_4)

    for i_4 in 1:n_MCTS_4
        j_4 = 0
        s_4 = SA[35,35]
        while !isterminal(m_4, s_4) && j_4 < n_steps_4
            a_4 = select_action(m_4, s_4)
            s_4, r_4 = @gen(:sp, :r)(m_4,s_4,a_4)
            reward_4[i_4] += discount(m_4)^(j_4-1) * r_4
            j_4 += 1
        end
    end

    println()
    println("            ### QUESTION 4: ###")
    println("mean reward:         |  ",mean(reward_4))
    println("sem reward:          |   ",std(reward_4)/sqrt(length(reward_4)))
    println()

end


############
# Question 5
############

function my_policy2(m, s)

    width = m.size[1]
    height = m.size[2]

    x = s[1]
    y = s[2]

    # right = s[1] < 20

    if (x > width-20 && x > 20) || mod(x,20) < 11
        return :left
    elseif x < 20 || mod(x,20) > 10
        return :right
    elseif (y > height-20 && y > 20) || mod(y,20) < 11
        return :down
    else
        return :up
    end
end

function simulate2!(m, n, q, s, d, t=nothing)

    if d == 0 || isterminal(m,s)
        return rollout(m, my_policy2, s)
    end

    if !haskey(n, (s, first(actions(m))))
        for a in actions(m)
            n[(s, a)] = 0
            q[(s, a)] = 0.0
        end
        return rollout(m, my_policy2, s)
    end

    ns = sum(n[(s,a)] for a in actions(m))

    c = 200.0
    a = argmax(a->q[(s,a)] + c*bonus(n[(s,a)], ns), actions(m))

    sp, r = @gen(:sp, :r)(m,s,a)

    q_val = r + discount(m)*simulate2!(m, n, q, sp, d-1, t)

    n[(s, a)] += 1
    q[(s, a)] += (q_val-q[(s, a)])/n[(s, a)]
    t[(s, a, sp)] = get(t, (s, a, sp), 0) + 1
    
    return q_val

end

function select_action2(m, s)

    start = time_ns()

    S = statetype(m)
    A = actiontype(m)

    n = Dict{Tuple{S, A}, Int}()
    q = Dict{Tuple{S, A}, Float64}()
    t = Dict{Tuple{S, A, S}, Int}()

    d = 40
    n_sims = 2000

    i = 0

    while time_ns() < start + 40_000_000 && i < n_sims
        simulate2!(m, n, q, s, d, t)
        i += 1
    end

    return argmax(a->q[(s, a)], actions(m))    

end

HW3.evaluate(select_action, "mage7128@colorado.edu", time=true)