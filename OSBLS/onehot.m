function [ Y_onehot ] = onehot( Y, num_c )
    num_n = length(Y);
    Y_onehot = zeros(num_n,num_c);
    for i=1:num_n  
        j = Y(i,1);
        Y_onehot(i,j) = 1;
    end
end

