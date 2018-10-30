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
