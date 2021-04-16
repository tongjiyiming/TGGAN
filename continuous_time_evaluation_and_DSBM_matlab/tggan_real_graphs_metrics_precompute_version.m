%% real auth data
% real_graphs = table2array(readtable('auth_user_0.txt'));
% n_times = 10;
% n_nodes = 27;

%% real metro data
real_graphs = table2array(readtable('metro_user_4.txt'));
n_times = 24;
n_nodes = 91;

%% preprocessing
d_col = 1;
i_col = 2;
j_col = 3;
t_col = 4;
time_interval = 1. / n_times;
real_graphs(:, t_col) = ceil(real_graphs(:, t_col) / time_interval);
real_graphs(:, i_col) = real_graphs(:, i_col) + 1;
real_graphs(:, j_col) = real_graphs(:, j_col) + 1;
real_graphs(:, d_col) = real_graphs(:, d_col) - min(real_graphs(:, d_col)) + 1;
d = max(real_graphs(:, d_col));

%% metrics computing
tic();
precompute_graphs = cell(1, 1);
precompute_betweenness = cell(1, 1);

agg_betweenness = zeros([d, n_nodes]);

for i = 1:d
    ind = real_graphs(:, d_col) == i;
    graph = real_graphs(ind, :);
    graph = graph(:, [i_col, j_col, t_col]);
    
    % avoid recompute the same graph again
    isPrecomputed = 0;
    if i > 1
        for j = 1:size(precompute_graphs)
            pre_graph = precompute_graphs{j};
            if size(graph, 1) == size(pre_graph, 1)
                n_same_edges = sum(graph == pre_graph, 'all');
                total_edges = size(graph, 1) * size(graph, 2);
                if n_same_edges == total_edges
                    isPrecomputed = 1;
                    k = j;
                end
            end
        end
    end
    
    if isPrecomputed == 0 && i == 1
        [ BC, dS, nF ] = betweennessCentrality(graph, n_nodes, n_times, 1);
        precompute_graphs{1, 1} = graph;
        precompute_betweenness{1, 1} = BC;
    elseif isPrecomputed == 0 && i > 1
        [ BC, dS, nF ] = betweennessCentrality(graph, n_nodes, n_times, 1);
        precompute_graphs{1, size(precompute_graphs, 2)+1} = graph;
        precompute_betweenness{1, size(precompute_graphs, 2)+1} = BC;
    else
        BC = precompute_betweenness{1, k};
    end
    %
    
%     [ BC, dS, nF ] = betweennessCentrality(graph, n_nodes, n_times, 1);
    
    agg_betweenness(i, :) = BC;
end
disp('end');
disp(sum(agg_betweenness, 1));
heatmap(agg_betweenness(randsample(d,100), :));
toc();

% % using a weight graph to put all sample graphs
% weighted_graph = zeros([n_nodes, n_nodes, n_times]);
% for i = 1:d
%     ind = real_graphs(:, d_col) == i;
%     graph = real_graphs(ind, :);
%     graph = graph(:, [i_col, j_col, t_col]);
%     weighted_graph = weighted_graph + networksFromContacts(graph, n_nodes, n_times, 1);
% end
% heatmap(sum(weighted_graph, 3));
% weighted_graph = weighted_graph / d;
% weighted_graph = arrayToContactSeq(weighted_graph, 1);
% disp(max(weighted_graph(:, 4)));
% 
% disp('start metrics computing');
% tic();
% [ betweennessCent, durationShortestPaths, nFastestPaths ] = betweennessCentrality(weighted_graph, n_nodes, n_times, 1);
% disp('betweennessCentrality');
% disp(betweennessCent);
% toc();