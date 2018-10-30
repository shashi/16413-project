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

function maximize(state, player)
    
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
    scores = [minimize(s[2], O) for s in succ]
    nexplr = sum(s[2] for s in scores) + 1
    val, idx = findmax(first.(scores))
    score, x, optim_play = scores[idx]
    return score, nexplr, succ[idx][1]
end

function minimize(state, player)
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
    scores = [maximize(s[2], X) for s in succ]
    nexplr = sum(s[2] for s in scores) + 1
    val, idx = findmin(first.(scores))
    score, x, optim_play = scores[idx]
    return score, nexplr, succ[idx][1]
end


######## The part below integrates the Optimal player with AlphaGo neural network player
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
