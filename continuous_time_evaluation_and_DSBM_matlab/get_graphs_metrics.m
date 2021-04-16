clc;
clear all;
close all;


% data_name = 'auth';
% data_name = 'metro';
% data_name = 'scale_free_nodes_100';
% data_name = 'scale_free_nodes_500';
data_name = 'scale_free_nodes_2500';
% data_name = 'protein100';
% data_name = 'auth_flow';
% data_name = 'metro_flow';

% model_name = 'tggan_';
% model_name = 'graphrnn_';
% model_name = 'graphvae_';
% model_name = 'dsbm_';
% model_name = 'netgan_';
% model_name = 'wenbin_';
model_name = 'taggen_';

% isTest = true;
isTest = false;

%% real graphs evaluation
real_metric_file = ['real_' data_name '_metrics.mat'];
if isfile(real_metric_file)
    load(real_metric_file);
    disp('load precomputed real graph metrics');
else
    [real_sample_betweenness,real_sample_closeness,real_sample_broadcastCentrality,...
        real_sample_receiveCentrality,real_sample_temporalCorrelation,real_sample_nodeTemporalCorrelation,...
        real_sample_temporalSmallWorldness,real_sample_burstiness] =...
        compute_metrics(['real_' data_name], isTest);
    save(real_metric_file, 'real_sample_betweenness', 'real_sample_closeness', 'real_sample_broadcastCentrality', ...
    'real_sample_receiveCentrality', 'real_sample_temporalCorrelation', 'real_sample_nodeTemporalCorrelation', ...
    'real_sample_temporalSmallWorldness', 'real_sample_burstiness')
end

%% evaluation
[model_sample_betweenness,model_sample_closeness,model_sample_broadcastCentrality,...
    model_sample_receiveCentrality,model_sample_temporalCorrelation,model_sample_nodeTemporalCorrelation,...
    model_sample_temporalSmallWorldness,model_sample_burstiness] =...
    compute_metrics([model_name data_name], isTest);
disp('model_sample_closeness.size()')
disp(size(model_sample_closeness))
disp('real_sample_closeness.size()')
disp(size(real_sample_closeness))
model_betweenness_mmd = mmd(real_sample_betweenness, model_sample_betweenness, 1);
model_closeness_mmd = mmd(real_sample_closeness, model_sample_closeness, 1);
model_broadcastCentrality_mmd = mmd(real_sample_broadcastCentrality, model_sample_broadcastCentrality, 1);
model_receiveCentrality_mmd = mmd(real_sample_receiveCentrality, model_sample_receiveCentrality, 1);
model_temporalCorrelation_mmd = mmd(real_sample_temporalCorrelation, model_sample_temporalCorrelation, 1);
model_nodeTemporalCorrelation_mmd = mmd(real_sample_nodeTemporalCorrelation, model_sample_nodeTemporalCorrelation, 1);
model_temporalSmallWorldness_mmd = mmd(real_sample_temporalSmallWorldness, model_sample_temporalSmallWorldness, 1);
model_burstiness_mmd = mmd(real_sample_burstiness, model_sample_burstiness, 1);

%% save the metrics
metric_file_name = [data_name '_' model_name 'all_metrics_' datestr(datetime(), 'mmm-dd-yyyy_HH-MM-SS') '.mat'];
if ~isTest
    save(metric_file_name, 'model_betweenness_mmd', 'model_closeness_mmd', 'model_broadcastCentrality_mmd', ...
    'model_receiveCentrality_mmd', 'model_temporalCorrelation_mmd', 'model_nodeTemporalCorrelation_mmd', ...
    'model_temporalSmallWorldness_mmd', 'model_burstiness_mmd')
end
% % using a weight graph to put all sample graphs
% weighted_graph = zeros([n_nodes, n_nodes, n_times]);
% for i = 1:d
%     ind = graphs(:, d_col) == i;
%     graph = graphs(ind, :);
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