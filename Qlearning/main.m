clear;
clc;

world  = 1;
gwinit(world);

global GWXSIZE;
global GWYSIZE;

positionsX = [];
positionsY = [];
actions = [];

%Q-matrix storing rewards for every action from every state/position.
Q = zeros(GWXSIZE, GWYSIZE, 4); %(posx, posy, action)

% Randomly initialize Q
%Q = rand(GWXSIZE, GWYSIZE, 4);


%%
tic
maxEpoch = 1000;
for epoch = 1:maxEpoch

    %Initialize Gridworld 1, 2, 3 or 4 and robot.
    gwinit(world);
    gamma = 0.7;
    alpha = 0.1;
    state = gwstate;

    while ~state.isterminal

        state = gwstate;

        %Choose between exploring or exploiting
        e = rand();
        threshold = 1 - 0.99 * epoch/maxEpoch;
        threshold = 0.9;
        if(e < threshold)
            %Random (exploring) action
            action = randi([1,4], 1, 1);
        else
            %Greedy (exploiting) action
            %choose the action with max value in Q
            [~, bestAction] = max(Q(state.pos(1), state.pos(2), :));
            action = bestAction;
        end

        % Save current state.
        currState = state;
        % Move to next state.
        gwaction(action);
        nextState = gwstate;
        % Calculate the max Q of next state-
        maxQnext = max(Q(nextState.pos(1), nextState.pos(2), :));
        % Calculate Q from current and next state.
        Qcurrent = Q(currState.pos(1), currState.pos(2),action);
        
        reward = nextState.feedback;
        
        % Give the action to try to walk through a wall infinite negative
        % feedback.
        if action == 1 && currState.pos(1) == GWXSIZE
           reward = -inf;
        elseif action == 2 && currState.pos(1) == 1
           reward = -inf;
        elseif action == 3 && currState.pos(2) == GWYSIZE
           reward = -inf;
        elseif action == 4 && currState.pos(2) == 1
           reward = -inf;
        end
        
        % Set the reward for the terminal state.
        if nextState.isterminal
            reward = 10;
        end
        
        % Update Q
        Q(currState.pos(1), currState.pos(2),action) = (1 - alpha) * Qcurrent + alpha * (reward + gamma * maxQnext);

    end
    
    epoch

end
toc

%% Test the robot with the trained Q-matrix, without changing it.
positionsX = [];
positionsY = [];
actions = [];

gwinit(world);
state = gwstate;
while ~state.isterminal
    
        gwdraw;
        
        state = gwstate;        

        %choose the action with max value in Q
        [~, bestAction] = max(Q(state.pos(1), state.pos(2), :));
        action = bestAction;
        
        gwaction(action);
        
        % Save array with arrows (if not at terminal state)
        if ~state.isterminal
            positionsX = [positionsX, state.pos(1)];
            positionsY = [positionsY, state.pos(2)];
            actions = [actions, action];
        end
end

% Plot the path of the robot.
gwdraw;
for i = 1:length(actions)
    gwplotarrow([positionsX(i), positionsY(i)], actions(i));
end

%% Plot the max values of the Q-matrix, i.e. the most rewarded action in each state.

Qplot = zeros(GWXSIZE, GWYSIZE);
for x = 1:GWXSIZE
    for y = 1:GWYSIZE
        [~, action] = max(Q(x,y,:));
        Qplot(x,y) = action;
    end
end

gwdraw;
gwplotallarrows(Qplot);
