using WebIO, InteractBase, JSExpr, CSSUtil, Observables

const btn_style = Dict("width"=> "36px", "height"=>"36px", "borderRadius" => "4px", "margin"=>"2px")

function board()
    scope = Scope()
    click = scope["clicks"] = Observable{Any}(nothing)
    scope["notif"] = Observable("")
    for i=1:3, j=1:3
        scope["cell-$i-$j"] = Observable(" ")
    end
    btns = [button(label=scope["cell-$i-$j"], style=btn_style,
        events=Dict("click" => @js () -> $click[] = $((i,j))))
              for i=1:3, j=1:3]
    grid = hbox(mapslices((x...)->vbox(x...), reshape(btns, (3,3)), dims=1)...)
    scope.dom = hbox(grid, hskip(1em), scope["notif"])(alignitems("center"))
    scope
end

function makechan(ob)
    c = Channel{Any}(0)
    on(ob) do x
        @async put!(c, x)
    end
    c
end

# A function to play against a Human-like player
# (those familiar with Golang can see that we use human_moves as a channel, and the play function
# itself is called as a goroutine by play_with function.)

import AlphaGo: GameEnv, MCTSPlayer, initialize_game!, N, tree_search!, is_done

# Play against a human player, nn is the neural network
# human_moves: a Channel which gets populated by moves from the human
# accepted moves: an observable where this function writes valid moves (so the UI can update)
function _play(env::AlphaGo.GameEnv, nn, human_moves, accepted_moves,
              notif; tower_height = 6, num_readouts = 800, mode = 0)

  @assert 0 ≤ tower_height ≤ 19

  az = MCTSPlayer(env, nn, num_readouts = num_readouts, two_player_mode = true)

  initialize_game!(az)
  num_moves = 0

  mode = mode == 0 ? mode : 1
  while !is_done(az)
    if num_moves % 2 == mode
      notif[] = "Your turn"
      mv = take!(human_moves)
      move = Tuple(mv)
    else
      notif[] = "AlphaZero's turn"
      yield
      current_readouts = N(az.root)
      readouts = az.num_readouts

      while N(az.root) < current_readouts + readouts
        tree_search!(az)
      end

      move = pick_move(az)
      #println(to_kgs(move, az.env))
    end
    if play_move!(az, move)
      accepted_moves[] = (move, num_moves % 2)
      num_moves += 1
    end
  end

  #println(az.root.position)

  w = winner(State(az.position.board))
  notif[] = w == X ? "X wins." : w == O ? "O wins." : "It's a draw"
end

# Play with a specific neural network
function play_with(nn::NeuralNet)
    b = board()

    moves = Observable{Any}(nothing)
    on(moves) do (mv, mode,)
        i, j = mv
        b["cell-$i-$j"][] = mode==0 ? "X" : "O"
    end

    game = AlphaGo.GomokuEnv(3,3)
    t= @async _play(game, nn, makechan(b["clicks"]), moves, b["notif"])

    b,t
end

