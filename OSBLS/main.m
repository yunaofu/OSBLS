clear; clc;

data_root = 'data';
addpath(data_root);

opts.N1 = 11;             
opts.Ng = 10;             
opts.N2 = 140;            
opts.s  = 0.8;            
opts.c  = 4;              
opts.maxIter = 30;
opts.pretreatMode = 4;
opts.seed = 106;
opts.alphaLib  = 2 .^ -1;
opts.betaLib   = 2 .^ -3;
opts.lambdaLib = 2 .^ -1;

% opts.N1 = 13;             
% opts.Ng = 10;             
% opts.N2 = 140;            
% opts.s  = 0.8;            
% opts.c  = 4;              
% opts.maxIter = 30;
% opts.pretreatMode = 1;
% opts.seed = 129;
% opts.alphaLib  = 2 .^ 1;
% opts.betaLib   = 2 .^ -3;
% opts.lambdaLib = 2 .^ 1;

for k = 1
    files = dir(fullfile(data_root, sprintf('SUB%d*.mat', k)));
    if isempty(files)
        error('File not found: SUB%d*.mat (directory: %s)', k, data_root);
    end

    S   = load(fullfile(files(1).folder, files(1).name));
    fn  = fieldnames(S);
    SUB = S.(fn{1});

    best_acc = OSBLS(SUB.fea_session1, SUB.gnd_session1, SUB.fea_session3, SUB.gnd_session3, opts);

    fprintf('Subject %d: BEST_ACC = %.4f\n', k, best_acc);
end