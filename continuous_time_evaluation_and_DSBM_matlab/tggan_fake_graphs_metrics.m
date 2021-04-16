clc;
clear all;
close all;

%% auth data
load('fake_graph_TGGAN_auth_iter_30000.mat');
n_times = 4;
n_nodes = 27;
fake_graphs = fake_graphs(1:3, :, :, :);
time_interval = 1. / n_times;
[d1, d2, d3, d4] = size(fake_graphs);
d_size = d1*d2;
fake_graphs = reshape(fake_graphs, [d_size, d3, 3]);
graphs = [];
for d = 1:d_size
    one_graph = reshape(fake_graphs(d, :, :), [d3, 3]);
    one_graph(:, 3) = 1 - one_graph(:, 3);
    ind = one_graph(:, 1) > -1;
    one_graph = one_graph(ind, :);
    one_graph = [repmat(d, size(one_graph, 1), 1) one_graph];
    graphs = [graphs; one_graph];
end

d_col = 1;
i_col = 2;
j_col = 3;
t_col = 4;
directed = 1;
time_interval = 1. / n_times;
graphs(:, d_col) = graphs(:, d_col) - min(graphs(:, d_col)) + 1;
graphs(:, i_col) = graphs(:, i_col) + 1;
graphs(:, j_col) = graphs(:, j_col) + 1;
graphs(:, t_col) = ceil(graphs(:, t_col) / time_interval);
unique_d = unique(graphs(:, d_col));

% %% metro data
% load('fake_graph_TGGAN_metro_iter_71000.mat');
% n_times = 6;
% n_nodes = 91;
% fake_graphs = fake_graphs(1:10, :, :, :);
% time_interval = 1. / n_times;
% [d1, d2, d3, d4] = size(fake_graphs);
% d_size = d1*d2;
% fake_graphs = reshape(fake_graphs, [d_size, d3, 3]);
% graphs = [];
% for d = 1:d_size
%     one_graph = reshape(fake_graphs(d, :, :), [d3, 3]);
%     one_graph(:, 3) = 1 - one_graph(:, 3);
%     ind = one_graph(:, 1) > -1;
%     one_graph = one_graph(ind, :);
%     one_graph = [repmat(d, size(one_graph, 1), 1) one_graph];
%     graphs = [graphs; one_graph];
% end
% 
% d_col = 1;
% i_col = 2;
% j_col = 3;
% t_col = 4;
% directed = 1;
% time_interval = 1. / n_times;
% graphs(:, d_col) = graphs(:, d_col) - min(graphs(:, d_col)) + 1;
% graphs(:, i_col) = graphs(:, i_col) + 1;
% graphs(:, j_col) = graphs(:, j_col) + 1;
% graphs(:, t_col) = ceil(graphs(:, t_col) / time_interval);
% unique_d = unique(graphs(:, d_col));

%% metrics computing
disp('start metrics computing');
unique_graphs = cell(1);
unique_graphs_inx = cell(1);

% avoid recompute the same graph again
for i = 1:d_size
% for i = 1:1
    d = unique_d(i);
    ind = graphs(:, d_col) == d;
    graph = graphs(ind, :);
    graph = graph(:, [i_col, j_col, t_col]);
    graph = unique(graph, 'rows');
    
    isExist = 0;
    k = 0;
    if i > 1
        for j = 1:size(unique_graphs, 2)
            pre_graph = unique_graphs{j};
            if size(graph, 1) == size(pre_graph, 1)
                n_same_elements = sum(graph == pre_graph, 'all');
                total_edges = size(graph, 1);
                if n_same_elements == total_edges * 3
                    isExist = 1;
                    k = j;
                    break;
                end
            end
        end
    end
    
    if i == 1
        unique_graphs{1} = graph;
        unique_graphs_inx{1} = [i];
    end
    if isExist == 0 && i > 1
        len = size(unique_graphs, 2);
        unique_graphs{len+1} = graph;
        unique_graphs_inx{len+1} = [i];
    end
    if isExist == 1 && i > 1
        find_inx = unique_graphs_inx{k};
        unique_graphs_inx{k} = [find_inx i];
    end
end

%% compute metrics for unique graphs
unique_n_graphs = size(unique_graphs, 2);
unique_sample_betweenness = zeros([unique_n_graphs, n_nodes]);
unique_sample_closeness = zeros([unique_n_graphs, n_nodes*(n_times-1)]);
unique_sample_broadcastCentrality = zeros([unique_n_graphs, n_nodes]);
unique_sample_receiveCentrality = zeros([unique_n_graphs, n_nodes]);
unique_sample_temporalCorrelation = zeros([unique_n_graphs, 1]);
unique_sample_nodeTemporalCorrelation = zeros([unique_n_graphs, n_nodes]);
unique_sample_temporalSmallWorldness = zeros([unique_n_graphs, 1]);
unique_sample_burstiness = zeros([unique_n_graphs, 1]);

now1 = tic();
parfor i = 1:unique_n_graphs
% for i = 1:1
%     disp(i);
    graph = unique_graphs{i};
    raw_graph_inx = unique_graphs_inx{i};
    
    %% betweenness centrality
    [ BC, dS, nF ] = betweennessCentrality(graph, n_nodes, n_times, directed);
    unique_sample_betweenness(i, :) = BC;
    if sum(unique_sample_betweenness(i, :)) ~= sum(BC)
        disp('betweennessCentrality error!');
        disp(i);
    end
    
    %% closeness centrality
    closeness = zeros([n_nodes, n_times-1]);
    for k = 1:n_nodes
        for t = 1:n_times-1
            [ Cc, tau_vec ] = closenessCentrality(k, t, graph, n_nodes, n_times, directed);
            closeness(k, t) = Cc;
        end
    end
    closeness = reshape(closeness, [1, n_nodes*(n_times-1)]);
    unique_sample_closeness(i, :) = closeness;
    if sum(unique_sample_closeness(i, :)) ~= sum(closeness)
        disp('closeness error!');
        disp(i);
    end
    
    %% broadcastRecieveCentrality
    alpha = 0.5;
    [ broadcastCentrality, receiveCentrality ] = broadcastRecieveCentrality( graph,alpha,n_nodes,n_times );
    unique_sample_broadcastCentrality(i, :) = broadcastCentrality;
%     disp(['broadcastCentrality ' mat2str(broadcastCentrality)]);
    unique_sample_receiveCentrality(i, :) = receiveCentrality;
    if sum(unique_sample_broadcastCentrality(i, :)) ~= sum(broadcastCentrality) && ~isnan(sum(broadcastCentrality))
        disp('broadcastCentrality error!');
        disp(sum(unique_sample_broadcastCentrality(i, :)));
        disp(sum(broadcastCentrality));
        disp(i);
    end
    if sum(unique_sample_receiveCentrality(i, :)) ~= sum(receiveCentrality) && ~isnan(sum(receiveCentrality))
        disp('receiveCentrality error!');
        disp(i);
    end
    
    %% temporal correlation
    [ C,C_vec ] = temporalCorrelation( graph,n_nodes,n_times,directed );
    unique_sample_temporalCorrelation(i, :) = C;
    unique_sample_nodeTemporalCorrelation(i, :) = C_vec;
    if sum(unique_sample_temporalCorrelation(i, :)) ~= sum(C)
        disp('temporalCorrelation error!');
        disp(i);
    end
    if sum(unique_sample_nodeTemporalCorrelation(i, :)) ~= sum(C_vec)
        disp('nodeTemporalCorrelation error!');
        disp(i);
    end
    
    %% temporalSmallWorldness
    [ smallWorldness ] = temporalSmallWorldness(graph,n_nodes,n_times,C,nF,directed);
    unique_sample_temporalSmallWorldness(i, :) = smallWorldness;
    if sum(unique_sample_temporalSmallWorldness(i, :)) ~= sum(smallWorldness) && ~isnan(sum(smallWorldness))
        disp('temporalSmallWorldness error!');
        disp(['temporalSmallWorldness ' mat2str(sum(smallWorldness, [1, 2]))]);
        disp(i);
    end
    if smallWorldness < 0, disp('smallWorldness is less than 0'); end
    
    %% burstiness
    [ B, cov, sigma, m ] = burstiness(graph);
    unique_sample_burstiness(i, :) = B;
    if sum(unique_sample_burstiness(i, :)) ~= sum(B)
        disp('burstiness error!');
        disp(i);
    end
    
end

sample_betweenness = zeros([d_size, n_nodes]);
sample_closeness = zeros([d_size, n_nodes*(n_times-1)]);
sample_broadcastCentrality = zeros([d_size, n_nodes]);
sample_receiveCentrality = zeros([d_size, n_nodes]);
sample_temporalCorrelation = zeros([d_size, 1]);
sample_nodeTemporalCorrelation = zeros([d_size, n_nodes]);
sample_temporalSmallWorldness = zeros([d_size, 1]);
sample_burstiness = zeros([d_size, 1]);
for i = 1:unique_n_graphs
    raw_graph_inx = unique_graphs_inx{i};
    
    BC = unique_sample_betweenness(i, :);
    closeness = unique_sample_closeness(i, :);
    broadcastCentrality = unique_sample_broadcastCentrality(i, :);
    receiveCentrality = unique_sample_receiveCentrality(i, :);
    C = unique_sample_temporalCorrelation(i, :);
    C_vec = unique_sample_nodeTemporalCorrelation(i, :);
    smallWorldness = unique_sample_temporalSmallWorldness(i, :);
    B = unique_sample_burstiness(i, :);
    
    for j = raw_graph_inx
        sample_betweenness(j, :) = BC;
        sample_closeness(j, :) = closeness;
        sample_broadcastCentrality(j, :) = broadcastCentrality;
        sample_receiveCentrality(j, :) = receiveCentrality;
        sample_temporalCorrelation(j, :) = C;
        sample_nodeTemporalCorrelation(j, :) = C_vec;
        sample_temporalSmallWorldness(j, :) = smallWorldness;
        sample_burstiness(j, :) = B;
    end
    
%     sample_betweenness(raw_graph_inx, :) = repmat(BC, size(raw_graph_inx, 2), 1);
%     sample_closeness(raw_graph_inx, :) = repmat(closeness, size(raw_graph_inx, 2), 1);
end

run_time = toc(now1);
disp('running time');
disp(run_time);
disp('end');

disp(['sample_betweenness ' mat2str(sum(sample_betweenness, [1, 2]))]);
disp(['sample_closeness ' mat2str(sum(sample_closeness, [1, 2]))]);
disp(['sample_broadcastCentrality ' mat2str(sum(sample_broadcastCentrality, [1, 2]))]);
disp(['sample_receiveCentrality ' mat2str(sum(sample_receiveCentrality, [1, 2]))]);
disp(['sample_temporalCorrelation ' mat2str(sum(sample_temporalCorrelation, [1, 2]))]);
disp(['sample_nodeTemporalCorrelation ' mat2str(sum(sample_nodeTemporalCorrelation, [1, 2]))]);
disp(['sample_temporalSmallWorldness ' mat2str(sum(sample_temporalSmallWorldness, [1, 2]))]);
disp(['sample_burstiness ' mat2str(sum(sample_burstiness, [1, 2]))]);

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