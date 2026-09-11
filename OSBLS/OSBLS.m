function best_acc = OSBLS(X_label, Y_label, X_unlabel, Y_unlabel, opts)

N1 = opts.N1;   Ng = opts.Ng;   N2 = opts.N2;
s  = opts.s;    c  = opts.c;    maxIter = opts.maxIter;
alphaLib = opts.alphaLib; betaLib = opts.betaLib; lambdaLib = opts.lambdaLib;
rng(opts.seed);

[~, X_label, Y_label, X_unlabel, Y_unlabel] = process_data(X_label', Y_label, X_unlabel', Y_unlabel, opts.pretreatMode);

n_L = size(X_label, 2);
n_U = size(X_unlabel, 2);
Y_unlabel = Y_unlabel(:);

train_x = zscore(X_label)';
test_x  = zscore(X_unlabel)';
Ns = N1 * Ng;

%% Feature nodes
X1 = [train_x, 0.1 * ones(size(train_x, 1), 1)];
feature_nodes = zeros(size(train_x, 1), Ns);
We = cell(1, Ng);
ps = cell(1, Ng);

for i = 1:Ng
    Wr = 2 * rand(size(train_x, 2) + 1, N1) - 1;
    A1 = mapminmax(X1 * Wr);

    Ws = sparse_bls(A1, X1)';
    We{i} = Ws;

    [F1, ps1] = mapminmax((X1 * Ws)', 0, 1);
    ps{i} = ps1;
    feature_nodes(:, N1*(i-1)+1 : N1*i) = F1';
end
clear X1 A1 Wr F1;

%% Enhancement nodes
X2 = [feature_nodes, 0.1 * ones(size(feature_nodes, 1), 1)];
if Ns >= N2
    wh = orth(2 * rand(Ns + 1, N2) - 1);
else
    wh = orth(2 * rand(Ns + 1, N2)' - 1)';
end
enhance = X2 * wh;
L2 = s / max(max(enhance));
AX = [feature_nodes, tansig(enhance * L2)];
clear X2 enhance;

XX1 = [test_x, 0.1 * ones(size(test_x, 1), 1)];
feature_nodes_test = zeros(size(test_x, 1), Ns);
for i = 1:Ng
    F2 = mapminmax('apply', (XX1 * We{i})', ps{i})';
    feature_nodes_test(:, N1*(i-1)+1 : N1*i) = F2;
end
XX2 = [feature_nodes_test, 0.1 * ones(size(feature_nodes_test, 1), 1)];
AX_test = [feature_nodes_test, tansig(XX2 * wh * L2)];
clear XX1 XX2 F2 wh We ps L2 train_x test_x feature_nodes feature_nodes_test;

A = [AX; AX_test]';
clear AX AX_test;

[d, n] = size(A);
Y_L_onehot = onehot(Y_label, c);

best_acc = 0;

for ia = 1:numel(alphaLib)
    alpha = alphaLib(ia);
for ib = 1:numel(betaLib)
    beta = betaLib(ib);
for il = 1:numel(lambdaLib)
    lambda = lambdaLib(il);

    Y = ones(n, c) / c;
    Y(1:n_L, :) = Y_L_onehot;
    P = 2 * Y - 1;
    P(n_L+1:n, :) = 0;
    S = zeros(n, c);

    [W, ~, ~] = svd(rand(d, c), 'econ');
    H = A' * W;
    J_prev = inf;

    for iter = 1:maxIter
        R = Y + P .* S;
        [~, pseudo] = max(R, [], 2);

        m_global = mean(H, 1);
        M_mat = repmat(m_global, n, 1);
        M_hat = zeros(n, c);
        for k = 1:c
            idx = (pseudo == k);
            if any(idx)
                M_hat(idx, :) = repmat(mean(H(idx, :), 1), nnz(idx), 1);
            end
        end

        % H-step
        H = (A'*W + alpha*R - beta*M_mat + 2*beta*M_hat) / (1 + alpha + 2*beta);

        % W-step
        D = diag(0.5 ./ sqrt(sum(W .* W, 2) + eps));
        W = GPI(A * A' + lambda * D, A * H);

        % Y-step
        for idx = n_L+1:n
            Y(idx, :) = EProjSimplex_new(H(idx, :) - P(idx, :) .* S(idx, :));
        end
        P = 2 * Y - 1;

        % S-step
        S = zeros(n, c);
        mask = abs(P) > eps;
        S(mask) = max(0, (H(mask) - Y(mask)) ./ P(mask));

        R = Y + P .* S;
        [~, pseudo] = max(R, [], 2);
        m_global = mean(H, 1);

        term1 = norm(A' * W - H, 'fro')^2;
        term2 = norm(H - R, 'fro')^2;
        term3 = sum(sqrt(sum(W .* W, 2) + eps));
        term4 = norm(H, 'fro')^2;
        for k = 1:c
            idx = (pseudo == k);
            n_k = nnz(idx);
            if n_k > 0
                H_k = H(idx, :);
                M_k = repmat(mean(H_k, 1), n_k, 1);
                term4 = term4 + norm(H_k - M_k, 'fro')^2 - norm(M_k - repmat(m_global, n_k, 1), 'fro')^2;
            end
        end
        OBJ = term1 + alpha*term2 + lambda*term3 + beta*term4;

        [~, predict_label] = max(R(n_L+1:n, :), [], 2);
        acc = nnz(predict_label == Y_unlabel) / n_U;
        if acc > best_acc
            best_acc = acc;
        end

        if abs(OBJ - J_prev) < 1e-3
            break;
        end
        J_prev = OBJ;
    end

end
end
end

end

function wk = sparse_bls(A, b)
lam = 1e-3; itrs = 50;
m = size(A, 2);
wk = zeros(m, size(b, 2));
ok = wk; uk = wk;
L1 = eye(m) / (A' * A + eye(m));
L2 = L1 * A' * b;
for i = 1:itrs
    ck = L2 + L1 * (ok - uk);
    ok = max(ck + uk - lam, 0) - max(-(ck + uk) - lam, 0);
    uk = uk + ck - ok;
    wk = ok;
end
end