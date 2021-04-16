
%% auth data
load('20191028-222439_assembled_graph_iter_23000.mat');
n_times = 10;
n_nodes = 28;
graphs = fake_graphs;
time_interval = 1. / n_times;
[d1, d2, d3, d4] = size(graphs);
graphs = reshape(graphs, [d1*d2, d3, 3]);
d = d1*d2;

%% metro data
% load('20191103-235308_assembled_graph_iter_71000.mat');
% n_times = 4;
% n_nodes = 91;
% graphs = fake_graphs;
% time_interval = 1. / n_times;
% [d1, d2, d3, d4] = size(graphs);
% graphs = reshape(graphs, [d1*d2, d3, 3]);
% d = d1*d2;

%% metrics computing
tic();
disp('start metrics computing');
agg_betweenness = zeros([d, n_nodes]);

% for i = 94:94
for i = 1:d
    graph = graphs(i, :, :);
    graph = reshape(graph, [d3, d4]);
    ind = graph(:, 1) > -1;
    graph = graph(ind, :);
    % descretize time
    graph(:, 3) = ceil(graph(:, 3) / time_interval);
    % nodes index start from zero, matlab should start from 1
    graph(:, 1) = graph(:, 1) + 1;
    graph(:, 2) = graph(:, 2) + 1;
    
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
    
%     [ BC, dS, nF ] = betweennessCentrality(graph, n_nodes, n_times, 1);
    agg_betweenness(i, :) = BC;
end
disp('end');
disp(sum(agg_betweenness, 1));
heatmap(agg_betweenness(randsample(d,100), :));
toc();

% % using a weight graph to put all sample graphs
% weighted_graph = zeros([n_nodes, n_nodes, n_times]);
% % for i = 94:94
% for i = 1:d
%     graph = graphs(i, :, :);
%     graph = reshape(graph, [d3, d4]);
%     ind = graph(:, 1) > -1;
%     graph = graph(ind, :);
%     % descretize time
%     graph(:, 3) = ceil(graph(:, 3) / time_interval);
%     % nodes index start from zero, matlab should start from 1
%     graph(:, 1) = graph(:, 1) + 1;
%     graph(:, 2) = graph(:, 2) + 1;
%     weighted_graph = weighted_graph + networksFromContacts(graph, n_nodes, n_times, 1);
% end
% 
% heatmap(sum(weighted_graph, 3));
% weighted_graph = weighted_graph / d;
% weighted_graph = arrayToContactSeq(weighted_graph, 1);
% 
% tic();
% [ betweennessCent, durationShortestPaths, nFastestPaths ] = betweennessCentrality(weighted_graph, n_nodes, n_times, 1);
% disp('betweennessCentrality');
% disp(betweennessCent);
% toc();

% tic();
% % notice last time step is meaningless
% closeness = zeros([n_nodes, n_times-1]);
% for i = 1:n_nodes
%     for t = 1:n_times-1
%         [ Cc, tau_vec ] = closenessCentrality(i, t, weighted_graph, 1, n_times, n_nodes);
%         closeness(i, t) = Cc;
%     end
% end
% disp('closenessCentrality');
% disp(closeness);
% toc();