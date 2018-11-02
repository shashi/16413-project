# Neural network f(θ): (p, v) = f(θ)(board)
#   p is such that pₐ = Pr(a|board) -- use this probability to sample trees
#   v -- probability of current player winning from current board state

using StaticArrays

const E = Int8(0)
const X = Int8(1)
const O = Int8(-1)
const DRAW = Int8(3)

const State = SArray{Tuple{3, 3}, Int8, 2, 9}
const Edge = Tuple{State, State}
const start_state = State(zeros(Int8, 3,3))

move(s, i,j, play) = (@assert s[i, j] == 0; setindex(s, play, i, j))
next_states(s, player) = [((i,j), move(s, i,j, player))
                          for i=1:3, j=1:3 if s[i,j] == 0]

allequal(s) = length(s) <= 1 ? true : s[1] == s[2] && allequal(s[2:end])

function winner(s)
    for i=1:3
        if allequal(s[:,i]) && s[1,i] in (O,X)
            return s[1,i]
        end
        if allequal(s[i,:]) && s[1,i] in (O,X)
            return s[i,1]
        end
    end
    diag1 = map(i->s[i,i], (1,2,3))
    diag2 = map(i->s[i,4-i], (1,2,3))
    allequal(diag1) && diag1[1] in (O,X) && return diag1[1]
    allequal(diag2) && diag2[1] in (O,X) && return diag2[1]
    return E
end

using Memoize

@memoize function maximize(state, player, f=minimize)
    
    if player != X
        error("Wrong player")
    end

    w = winner(state)
    if w == E && all(!iszero, state)
        return 0, 1, state # draw
    elseif w != E
        return (w == X ? 1 : -1), 1, state
    end
    succ = next_states(state, X)
    scores = [f(s[2], O) for s in succ]
    nexplr = sum(s[2] for s in scores) + 1
    val, idx = findmax(first.(scores))
    score, x, optim_play = scores[idx]
    return score, nexplr, succ[idx][1]
end

@memoize function minimize(state, player, f=maximize)
    if player != O
        error("Wrong player")
    end
    
    w = winner(state)
    if w == E && all(!iszero, state)
        return 0, 1, state # draw
    elseif w != E
        return (w == X ? 1 : -1), 1, state
    end
    succ = next_states(state, O)
    scores = [f(s[2], X) for s in succ]
    nexplr = sum(s[2] for s in scores) + 1
    val, idx = findmin(first.(scores))
    score, x, optim_play = scores[idx]
    return score, nexplr, succ[idx][1]
end

function randomplay(state, player)
    w = winner(state)
    if w == E && all(!iszero, state)
        return 0, [state] # draw
    elseif w != E
        return (w == X ? 1 : -1), [state]
    end
    succ = next_states(state, player)
    score, states = randomplay(rand(succ)[2], player == X ? O : X)
    return score, vcat([state], states)
end

function update_stats!(stats, trace, score)
    for s in trace[2:end] # leave out start state
        # for each state we keep a sum of scores and the
        # number of times the state was reached
        sum, count = haskey(stats, s) ? stats[s] : (0, 0)
        sum += score
        count += 1
        stats[s] = (sum, count)
    end
end

function random_stats(f, n)
    stats = Dict{State, Tuple{Int, Int}}()
    # we store sum and number of times chosen
    for i=1:n
        score, trace = f(start_state, X) # run a random play -- gives score and a state
        update_stats!(stats, trace, score)
    end
    stats
end

# Upper Confidence Bound 1 applied to trees
function uct(w, n, nₚ)
    w/max(1, n) + (√2) * √(log(nₚ+1)/max(n, 1))
end

function mcts_move(state, player, priors)
    succ = next_states(state, player)
    prior_vec = [get(priors, s[2], (0,1)) for s in succ]
    _, nₚ = get(priors, state, (0,0)) # my count
    ns = last.(prior_vec); ws = first.(prior_vec)
    weights = uct.(ws, ns, nₚ)
    idx = sample(1:length(succ), Weights(weights), 1)[1]
    # sample a branch by weights
    succ[idx][1] # return the position
end

using StatsBase
function play_f(fₓ, fₒ, state, player=X; priors1=nothing, priors2=nothing, update=false)
    trace = State[]

    while true
        push!(trace, state)
        w = winner(state)
        if w == E && all(!iszero, state)
            update && update_stats!(priors1, trace, 0)
            return 0 # draw
        elseif w != E
            update && update_stats!(priors2, trace, w == X ? 1 : -1)
            return (w == X ? 1 : -1)
        end
        if player === X
            mv = fₓ(state, X)
        else
            mv = fₒ(state, O)
        end
        state = move(state, mv..., player)
        player = player === X ? O : X
    end
end

function play_mcts_vs_optimal(state, priors, player=X; update=false)
    play_f((s, pl) -> mcts_move(s, pl, priors),
         (s, pl) -> minimize(s, pl)[3], state,
         player, update=update, priors1=priors, priors2=priors)
end

function play_mcts_vs_mcts(state, priors1, priors2, player=X; update=false)
    play_f((s, pl) -> mcts_move(s, pl, priors1),
         (s, pl) -> mcts_move(s, pl, priors2),
         state, player, update=update,
         priors1=priors1, priors2=priors2)
end
 

######## The part below integrates the Optimal player with AlphaGo neural network player
using AlphaGo
import AlphaGo: GameEnv, MCTSPlayer, initialize_game!, N, tree_search!, is_done
adapt_state(x) = State(x.board)

function play_optimal(env::AlphaGo.GameEnv, nn; tower_height = 6, num_readouts = 800, mode=0)
  @assert 0 ≤ tower_height ≤ 19

  az = MCTSPlayer(env, nn, num_readouts = num_readouts, two_player_mode = true)

    states = []
  initialize_game!(az)
  num_moves = 0

  while !is_done(az)
        #println(az.root.position)
        push!(states,  copy(az.root.position.board))
    if num_moves % 2 == mode
        if num_moves == 0
            move = (rand(1:3), rand(1:3))
        else
            mv = maximize(adapt_state(az.root.position), 1)
            move = mv[3]
        end
    else
      current_readouts = N(az.root)
      readouts = az.num_readouts

      while N(az.root) < current_readouts + readouts
        tree_search!(az)
      end

      move = pick_move(az)
      #println(to_kgs(move, az.env))
    end
    if play_move!(az, move)
      num_moves += 1
    end
  end

  #println(az.root.position)
    winner(adapt_state(az.root.position))
end
