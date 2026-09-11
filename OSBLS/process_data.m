function [X, X_l, Y_l, X_u, Y_u] = process_data(X_src, Y_src, X_tar, Y_tar, mode)
if nargin < 5
    mode = 1;
end
[Y_sort, index] = sort(Y_src);
X_src = X_src(index, :);
Y_src = Y_sort;
[Y_sort, index] = sort(Y_tar);
X_tar = X_tar(index, :);
Y_tar = Y_sort;
[n_l, ~] = size(X_src);
[n_u, ~] = size(X_tar);
switch mode
    case 1
        X_l = X_src;
        [X_l, ~] = mapminmax(X_l', 0, 1);
        X_l = X_l';
        X_l = X_l - repmat(mean(X_l, 1), [n_l, 1]);
        Y_l = Y_src;
        X_u = X_tar;
        [X_u, ~] = mapminmax(X_u', 0, 1);
        X_u = X_u';
        X_u = X_u - repmat(mean(X_u, 1), [n_u, 1]);
        Y_u = Y_tar;
        X = [X_l; X_u];
        [n, ~] = size(X);
        X = X - repmat(mean(X, 1), [n, 1]);
    case 2
        X_l = X_src - repmat(mean(X_src, 1), [n_l, 1]);
        [X_l, ~] = mapminmax(X_l', 0, 1);
        X_l = X_l';
        Y_l = Y_src;
        X_u = X_tar - repmat(mean(X_tar, 1), [n_u, 1]);
        [X_u, ~] = mapminmax(X_u', 0, 1);
        X_u = X_u';
        Y_u = Y_tar;
        X = [X_l; X_u];
    case 3
        X_src = X_src - repmat(mean(X_src, 1), [n_l, 1]);
        X_tar = X_tar - repmat(mean(X_tar, 1), [n_u, 1]);
        X = [X_src; X_tar];
        [X, ~] = mapminmax(X', -1, 1);
        X = X';
        X_l = X(1:n_l, :);
        X_u = X(n_l+1:n_l+n_u, :);
        Y_l = Y_src;
        Y_u = Y_tar;
    case 4
        X_src = X_src - repmat(mean(X_src, 1), [n_l, 1]);
        X_tar = X_tar - repmat(mean(X_tar, 1), [n_u, 1]);
        X = [X_src; X_tar];
        X_l = X_src;
        X_u = X_tar;
        Y_l = Y_src;
        Y_u = Y_tar;
    otherwise
        error('process_data: unknown mode %d', mode);
end
X = X';
X_l = X_l';
X_u = X_u';
end