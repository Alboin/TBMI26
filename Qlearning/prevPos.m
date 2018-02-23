% Calculate a new position with given action.
function [x_out,y_out] = prevPos(x_in, y_in, action)

    global GWXSIZE;
    global GWYSIZE;

    x_out = x_in;
    y_out = y_in;

    if action == 1
        x_out = x_out - 1;
    elseif action == 2
        x_out = x_out + 1;
    elseif action == 3
        y_out = y_out - 1;
    elseif action == 4
        y_out = y_out + 1;
    end
    
    x_out = max(0, min(x_out, GWXSIZE));
    y_out = max(0, min(y_out, GWYSIZE));
end

