clear;
clc;

global GWXSIZE;
global GWYSIZE;
global GWTERM;

positionsX = [];
positionsY = [];
actions = [];

%Q = ones(GWXSIZE, GWYSIZE) * 3;

%Q-matrix storing rewards for every action from every state/position.
Q = zeros(GWXSIZE, GWYSIZE, 4); %(posx, posy, action)

R = zeros(GWXSIZE, GWYSIZE, 4);

% Set up the end state reward for R
[maxValue, linearIndexesOfMaxes] = max(GWTERM(:));
[rowsOfMaxes, colsOfMaxes] = find(GWTERM == maxValue);
for i = 1:4
    [x,y] = prevPos(rowsOfMaxes, colsOfMaxes, i);
    R(x,y,i) = 1; 
end

maxEpoch = 5000;
for epoch = 1:maxEpoch

    %Initialize Gridworld 1, 2, 3 or 4 and robot.
    gwinit(1);
    gamma = 0.7;
    alpha = 0.1;
    state = gwstate;

    while ~state.isterminal
        %gwdraw;
        %gwplotallarrows(Q);


        state = gwstate;

        %Choose between exploring or exploiting
        e = rand();
        threshold = 1 - 0.98 * epoch/maxEpoch;
        if(e < threshold)
            %Random (exploring) action
            action = round(rand() * 3 + 1);
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
        Q(currState.pos(1), currState.pos(2),action) = (1 - alpha) * Qcurrent + alpha * (R(currState.pos(1), currState.pos(2),action) + gamma * maxQnext);
        


    end
    
    epoch

end

%% Test the robot with the trained Q-matrix, without changing it.
positionsX = [];
positionsY = [];
actions = [];

gwinit(1);
state = gwstate;
while ~state.isterminal
    
        gwdraw;
        state = gwstate;

        threshold = 0.9;
        if(e < threshold)
            %Random (exploring) action
            action = round(rand() * 3 + 1);
        else
            %Greedy (exploiting) action
            %choose the action with max value in Q
            [~, bestAction] = max(Q(state.pos(1), state.pos(2), :));
            action = bestAction;
        end
        
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
