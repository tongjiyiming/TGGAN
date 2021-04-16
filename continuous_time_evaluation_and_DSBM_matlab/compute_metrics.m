function [sample_betweenness,sample_closeness,sample_broadcastCentrality,...
    sample_receiveCentrality,sample_temporalCorrelation,sample_nodeTemporalCorrelation,...
    sample_temporalSmallWorldness,sample_burstiness] = compute_metrics(data_name, isTest)
%     data_name = 'real_auth';
%     isTest = false;
    %% real and tggan
    if (size(strfind(data_name, 'real'), 1) > 0 || size(strfind(data_name, 'tggan'), 1) > 0) ...
        && ~strcmp(data_name, 'real_protein100') && ~strcmp(data_name, 'real_metro_flow') ...
        && ~strcmp(data_name, 'real_auth_flow')
        if strcmp(data_name, 'real_auth')
            graphs = importdata('auth_user_0.txt');
            n_times = 4;
            n_nodes = 27;
        end
        if strcmp(data_name, 'tggan_auth')
            load('fake_graph_TGGAN_auth_iter_30000.mat');
            ind = fake_graphs(:, 1) < 1000;
            graphs = fake_graphs(ind, :);
            n_times = 4;
            n_nodes = 27;
        end
        if  strcmp(data_name, 'real_metro')
            graphs = importdata('metro_user_4.txt');
            n_times = 6;
            n_nodes = 91;
        end
        if  strcmp(data_name, 'tggan_metro')
            load('fake_graph_TGGAN_metro_iter_71000.mat');
            ind = fake_graphs(:, 1) < 1000;
            graphs = fake_graphs(ind, :);
            n_times = 6;
            n_nodes = 91;
        end
        if  strcmp(data_name, 'real_scale_free_nodes_100')
            graphs = importdata('data-scale-free-nodes-100-samples-200.txt');
            n_times = 5;
            n_nodes = 100;
        end
        if  strcmp(data_name, 'tggan_scale_free_nodes_100')
            load('fake_graph_TGGAN_scale-free-100_iter_3000.mat');
            ind = fake_graphs(:, 1) < 1000;
            graphs = fake_graphs(ind, :);
            n_times = 5;
            n_nodes = 100;
        end
        if  strcmp(data_name, 'real_scale_free_nodes_500')
            graphs = importdata('data-scale-free-nodes-500-samples-100.txt');
            n_times = 5;
            n_nodes = 500;
        end
        if  strcmp(data_name, 'tggan_scale_free_nodes_500')
            load('fake_graph_TGGAN_scale-free-500_iter_3000.mat');
            ind = fake_graphs(:, 1) < 1000;
            graphs = fake_graphs(ind, :);
            n_times = 5;
            n_nodes = 500;
        end
        if  strcmp(data_name, 'real_scale_free_nodes_2500')
            graphs = importdata('data-scale-free-nodes-2500-samples-100.txt');
            n_times = 5;
            n_nodes = 2500;
        end
        if  strcmp(data_name, 'tggan_scale_free_nodes_2500')
            load('fake_graph_TGGAN_scale-free-2500_iter_3000.mat');
            ind = fake_graphs(:, 1) < 1000;
            graphs = fake_graphs(ind, :);
            n_times = 5;
            n_nodes = 2500;
        end
        d_col = 1;
        i_col = 2;
        j_col = 3;
        t_col = 4;
        directed = 1;
        time_interval = 1. / n_times;
        graphs(:, t_col) = ceil(graphs(:, t_col) / time_interval);
        graphs(:, i_col) = graphs(:, i_col) + 1;
        graphs(:, j_col) = graphs(:, j_col) + 1;
        graphs(:, d_col) = graphs(:, d_col) - min(graphs(:, d_col)) + 1;
        unique_d = unique(graphs(:, d_col));
        d_size = size(unique_d, 1);
    end
    
    %% TagGen paper from KDD 2020: https://dl.acm.org/doi/abs/10.1145/3394486.3403082
    if size(strfind(data_name, 'taggen'), 1) > 0
         if strcmp(data_name, 'taggen_auth')
             graphs = importdata('chen_output/fake_graph_taggen_auth.txt');
             n_times = 4;
             n_nodes = 27;
             scale = 2;
         end
         if strcmp(data_name, 'taggen_metro')
             graphs = importdata('chen_output/fake_graph_taggen_metro.txt');
             n_times = 6;
             n_nodes = 91;
             scale = 1;
         end
         if strcmp(data_name, 'taggen_scale_free_nodes_100')
             graphs = importdata('chen_output/fake_graph_taggen_SF_100_200.txt');
%              ind = graphs(:, 1) < 300;
%              graphs = graphs(ind, :);
             n_times = 5;
             n_nodes = 100;
             scale = 4;
         end
         if strcmp(data_name, 'taggen_scale_free_nodes_500')
             graphs = importdata('chen_output/fake_graph_taggen_SF_500_100.txt');
             ind = graphs(:, 1) < 300;
             graphs = graphs(ind, :);
             n_times = 5;
             n_nodes = 500;
             scale = 4;
         end
         if strcmp(data_name, 'taggen_scale_free_nodes_2500')
             graphs = importdata('chen_output/fake_graph_taggen_SF_2500_100.txt');
             ind = graphs(:, 1) < 10;
             graphs = graphs(ind, :);
             n_times = 5;
             n_nodes = 2500;
             scale = 4;
         end
        d_col = 1;
        i_col = 2;
        j_col = 3;
        t_col = 4;
        directed = 1;
%         time_interval = 1. / n_times;
%         graphs(:, t_col) = ceil(graphs(:, t_col) / time_interval);
        graphs(:, t_col) = floor(graphs(:, t_col)/scale) + 1;
        graphs(:, i_col) = graphs(:, i_col) + 1;
        graphs(:, j_col) = graphs(:, j_col) + 1;
        graphs(:, d_col) = graphs(:, d_col) - min(graphs(:, d_col)) + 1;
        unique_d = unique(graphs(:, d_col));
        d_size = size(unique_d, 1);
    end
        
    %% Wenbin paper geneted auth metor data, real Wenbin protein data
    if size(strfind(data_name, 'wenbin'), 1) > 0 || strcmp(data_name, 'real_protein100') ...
            || strcmp(data_name, 'real_metro_flow') || strcmp(data_name, 'real_auth_flow')
        if strcmp(data_name, 'wenbin_metro_flow')
            graphs = importdata('fake_graph_wenbin_metro_flow_generated.txt');
            n_times = 24;
            n_nodes = 91;
        end
        if strcmp(data_name, 'wenbin_auth_flow')
            graphs = importdata('fake_graph_wenbin_auth_flow_generated.txt');
            n_times = 10;
            n_nodes = 9;
        end
        if strcmp(data_name, 'real_metro_flow')
            graphs = importdata('DSBM\metro_flows_edges_real.txt');
            n_times = 24;
            n_nodes = 91;
        end
        if strcmp(data_name, 'real_auth_flow')
            graphs = importdata('DSBM\auth_flows_edges_real.txt');
            n_times = 10;
            n_nodes = 9;
        end
        if strcmp(data_name, 'real_protein100')
            graphs = importdata('protein_100_real.txt');
            n_times = 100;
            n_nodes = 8;
        end
        if strcmp(data_name, 'wenbin_protein100')
            graphs = importdata('fake_graph_wenbin_protein100_user_generated.txt');
            n_times = 100;
            n_nodes = 8;
        end
        if strcmp(data_name, 'wenbin_auth')
            graphs = importdata('fake_graph_wenbin_auth_user_generated.txt');
            n_times = 4;
            n_nodes = 27;
        end
        if strcmp(data_name, 'wenbin_metro')
            graphs = importdata('fake_graph_wenbin_metro_user_generated.txt');
            n_times = 6;
            n_nodes = 91;
        end
        if strcmp(data_name, 'wenbin_scale_free_nodes_100')
            graphs = importdata('fake_graph_wenbin_sim_100.txt');
            n_times = 5;
            n_nodes = 100;
        end
        if strcmp(data_name, 'wenbin_scale_free_nodes_500')
            graphs = importdata('fake_graph_wenbin_sim_500.txt');
            n_times = 5;
            n_nodes = 500;
        end
        if strcmp(data_name, 'wenbin_scale_free_nodes_2500')
            graphs = importdata('fake_graph_wenbin_sim_2500.txt');
            n_times = 5;
            n_nodes = 2500;
        end
        d_col = 1;
        i_col = 2;
        j_col = 3;
        t_col = 4;
        directed = 1;
        graphs(:, i_col) = graphs(:, i_col) + 1;
        graphs(:, j_col) = graphs(:, j_col) + 1;
        graphs(:, d_col) = graphs(:, d_col) - min(graphs(:, d_col)) + 1;
        unique_d = unique(graphs(:, d_col));

        d_size = size(unique_d, 1);
    end
    
    %% GraphRNN GraphVAE NetGAN geneted auth data
    if size(strfind(data_name, 'graphrnn'), 1) > 0 || size(strfind(data_name, 'graphvae'), 1) > 0 || size(strfind(data_name, 'netgan'), 1) > 0
        if strcmp(data_name, 'graphrnn_auth')
            fake_graphs = importdata('GraphRNN_RNN_auth_epoch_3000.txt');
            n_times = 4;
            n_nodes = 27;
            n_samples = 2000;
        end
        if strcmp(data_name, 'graphvae_auth')
            fake_graphs = importdata('GraphRNN_VAE_conditional_auth_epoch_3000.txt');
            n_times = 4;
            n_nodes = 27;
            n_samples = 2000;
        end
        if strcmp(data_name, 'netgan_auth')
            fake_graphs = load('fake_graph_netgan_auth_node_27_sample_1000_times_4.mat');
            fake_graphs = fake_graphs.edges;
            n_times = 4;
            n_nodes = 27;
            n_samples = 1000;
        end
        if strcmp(data_name, 'graphrnn_metro')
            fake_graphs = importdata('GraphRNN_RNN_metro_epoch_3000.txt');
            n_times = 6;
            n_nodes = 91;
            n_samples = 1000;
        end
        if strcmp(data_name, 'graphvae_metro')
            fake_graphs = importdata('GraphRNN_VAE_conditional_metro_epoch_3000.txt');
            n_times = 6;
            n_nodes = 91;
            n_samples = 1000;
        end
        if strcmp(data_name, 'netgan_metro')
            fake_graphs = load('fake_graph_netgan_metro_node_91_sample_1000_times_6.mat');
            fake_graphs = fake_graphs.edges;
            n_times = 6;
            n_nodes = 91;
            n_samples = 1000;
        end
        if strcmp(data_name, 'graphrnn_scale_free_nodes_100')
            fake_graphs = importdata('GraphRNN_RNN_simulation_node_100_samples_200_epoch_3000.txt');
            n_times = 5;
            n_nodes = 100;
            n_samples = 1000;
        end
        if strcmp(data_name, 'graphvae_scale_free_nodes_100')
            fake_graphs = importdata('GraphRNN_VAE_simulation_node_100_samples_200_epoch_3000.txt');
            n_times = 5;
            n_nodes = 100;
            n_samples = 1000;
        end
        if strcmp(data_name, 'netgan_scale_free_nodes_100')
            fake_graphs = load('fake_graph_netgan_scale_free_node_100_sample_200_times_5.mat');
            fake_graphs = fake_graphs.edges;
            n_times = 5;
            n_nodes = 100;
            n_samples = 1000;
        end
        if strcmp(data_name, 'graphrnn_scale_free_nodes_500')
            fake_graphs = importdata('GraphRNN_RNN_simulation_node_500_samples_100_epoch_1000.txt');
            n_times = 5;
            n_nodes = 500;
            n_samples = 1000;
        end
        if strcmp(data_name, 'graphvae_scale_free_nodes_500')
            fake_graphs = importdata('GraphRNN_VAE_simulation_node_500_samples_100_epoch_3000.txt');
            n_times = 5;
            n_nodes = 500;
            n_samples = 1000;
        end
        if strcmp(data_name, 'netgan_scale_free_nodes_500')
            fake_graphs = load('fake_graph_netgan_scale_free_node_500_sample_100_times_5.mat');
            fake_graphs = fake_graphs.edges;
            n_times = 5;
            n_nodes = 500;
            n_samples = 1000;
        end
        if strcmp(data_name, 'graphrnn_scale_free_nodes_2500')
            fake_graphs = importdata('GraphRNN_RNN_simulation_node_2500_samples_100_epoch_3000.txt');
            n_times = 5;
            n_nodes = 2500;
            n_samples = 100;
        end
        if strcmp(data_name, 'netgan_scale_free_nodes_2500')
            fake_graphs = load('fake_graph_netgan_scale_free_node_2500_sample_100_times_5.mat');
            fake_graphs = fake_graphs.edges;
            n_times = 5;
            n_nodes = 2500;
            n_samples = 100;
        end
        if strcmp(data_name, 'netgan_protein100')
            fake_graphs = importdata('fake_graph_netgan_protein_node_8_sample_1000_times_100.txt');
            n_times = 100;
            n_nodes = 8;
            n_samples = 100;
        end
        if strcmp(data_name, 'graphrnn_protein100')
            fake_graphs = importdata('GraphRNN_RNN_protein_8_nodes_8_samples_300_times_100_4_32__all_times.txt');
            n_times = 100;
            n_nodes = 8;
            n_samples = 100;
        end
        if strcmp(data_name, 'graphvae_protein100')
            fake_graphs = importdata('GraphRNN_VAE_conditional_protein_8_nodes_8_samples_300_times_100_4_32__all_times.txt');
            n_times = 100;
            n_nodes = 8;
            n_samples = 100;
        end
        if strcmp(data_name, 'netgan_protein100')
            fake_graphs = importdata('fake_graph_netgan_protein_node_8_sample_1000_times_100.txt');
            n_times = 100;
            n_nodes = 8;
            n_samples = 100;
        end
        if strcmp(data_name, 'graphrnn_auth_flow')
            fake_graphs = importdata('GraphRNN_RNN_auth-flow_nodes_9_samples_260_times_10_4_32__all_times.txt');
            n_times = 10;
            n_nodes = 9;
            n_samples = 100;
        end
        if strcmp(data_name, 'netgan_auth_flow')
            fake_graphs = importdata('fake_graph_netgan_auth-flow_node_9_sample_1000_times_10.txt');
            n_times = 10;
            n_nodes = 9;
            n_samples = 1000;
        end
        if strcmp(data_name, 'graphvae_auth_flow')
            fake_graphs = importdata('GraphRNN_VAE_conditional_auth-flow_nodes_9_samples_260_times_10_4_32_all_times.txt');
            n_times = 10;
            n_nodes = 9;
            n_samples = 1000;
        end
        if strcmp(data_name, 'netgan_metro_flow')
            fake_graphs = importdata('fake_graph_netgan_metro-flow_node_91_sample_1000_times_24.txt');
            n_times = 24;
            n_nodes = 91;
            n_samples = 100;
        end
        if strcmp(data_name, 'graphrnn_metro_flow')
            fake_graphs = importdata('GraphRNN_RNN_metro-flow_nodes_91_samples_122_times_24_all_times.txt');
            n_times = 24;
            n_nodes = 91;
            n_samples = 100;
        end
        if strcmp(data_name, 'graphvae_metro_flow')
            fake_graphs = importdata('GraphRNN_VAE_conditional_metro-flow_nodes_91_samples_122_times_24_all_times.txt');
            n_times = 24;
            n_nodes = 91;
            n_samples = 100;
        end
        d_col = 1;
        i_col = 2;
        j_col = 3;
        t_col = 4;
        directed = 1;
        fake_graphs(:, i_col) = fake_graphs(:, i_col) + 1;
        fake_graphs(:, j_col) = fake_graphs(:, j_col) + 1;
        fake_graphs(:, t_col) = fake_graphs(:, t_col) + 1;
        
        tmp_graphs_cells = {};
        unique_d_cells = {};
        for t = 1:n_times
            tmp_graphs_cells{t} = fake_graphs(fake_graphs(:, t_col) == t, :);
            unique_d_cells{t} = unique(tmp_graphs_cells{t}(:, d_col));
        end

        graphs = [];
        for i = 1:n_samples
            d_graphs_cells = {};
            for t = 1:n_times
                unique_d = unique_d_cells{t};
                tmp_graphs = tmp_graphs_cells{t};
                d_graphs = [];
                if size(unique_d, 1) > 0
                    d = randsample(unique_d, 1, true);
                    d_graphs = tmp_graphs(tmp_graphs(:, d_col) == d, :);
                    d_graphs = [repmat(i, [size(d_graphs, 1), 1]) d_graphs(:, [i_col j_col t_col])];
                    d_graphs_cells{t} = d_graphs;
                end
                graphs = [graphs; d_graphs];
            end
        end
        
        unique_d = unique(graphs(:, d_col));
        d_size = size(unique_d, 1);
    end

    %% DSBM geneted data
    if size(strfind(data_name, 'dsbm'), 1) > 0
        if strcmp(data_name, 'dsbm_auth_flow')
            load('DSBM/DSBM_auth_flows_May-30-2020_14-12-25.mat');
            n_times = 10;
            n_nodes = 9;
            n_samples = 100;
        end
        if strcmp(data_name, 'dsbm_metro_flow')
            load('DSBM/DSBM_metro_flows_May-30-2020_14-35-03.mat');
            n_times = 24;
            n_nodes = 91;
            n_samples = 10;
        end
        if strcmp(data_name, 'dsbm_auth')
            load('DSBM/DSBM_auth_Jan-10-2020_17-33-49.mat');
            n_times = 4;
            n_nodes = 27;
            n_samples = 2000;
        end
        if strcmp(data_name, 'dsbm_metro')
            load('DSBM/DSBM_metro_Jan-10-2020_18-40-56.mat');
            n_times = 6;
            n_nodes = 91;
            n_samples = 20;
        end
        if strcmp(data_name, 'dsbm_scale_free_nodes_100')
            load('DSBM/DSBM_scale-free-nodes-100_Feb-05-2020_12-15-32.mat');
            n_times = 5;
            n_nodes = 100;
            n_samples = 20;
        end
        if strcmp(data_name, 'dsbm_protein100')
            load('DSBM/DSBM_protein100_May-26-2020_21-29-08.mat');
            n_times = 100;
            n_nodes = 8;
            n_samples = 1000;
        end
        d_col = 1;
        i_col = 2;
        j_col = 3;
        t_col = 4;
        directed = 1;
        
        fake_graphs = adjDsbm_fake_graphs;
%         fake_graphs = adjSbtm_fake_graphs;
        d_total = size(fake_graphs, 3) / n_times;
        fake_graphs = reshape(fake_graphs, [n_nodes, n_nodes, n_times, d_total]);
        fake_graphs = fake_graphs(:, :, :, 1:n_samples);
        ind = find(fake_graphs);
        [i, j, t, d] = ind2sub(size(fake_graphs), ind);
        graphs = [d i j t];
        
        unique_d = unique(graphs(:, d_col));
        d_size = size(unique_d, 1);
    end

    %% test purpose
    if isTest
        ind = graphs(:, d_col) < 10;
        graphs = graphs(ind, :);
        unique_d = unique(graphs(:, d_col));
        d_size = size(unique_d, 1);
    end
    
    %% metrics computing
    disp('start metrics computing');
    unique_graphs = cell(1);
    unique_graphs_inx = cell(1);

    % avoid recompute the same graph again
    for i = 1:d_size
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
    disp([data_name ': total number of graphs ' mat2str(unique_n_graphs)]);
    unique_sample_betweenness = zeros([unique_n_graphs, n_nodes]);
    unique_sample_closeness = zeros([unique_n_graphs, n_nodes*(n_times-1)]);
    unique_sample_broadcastCentrality = zeros([unique_n_graphs, n_nodes]);
    unique_sample_receiveCentrality = zeros([unique_n_graphs, n_nodes]);
    unique_sample_temporalCorrelation = zeros([unique_n_graphs, 1]);
    unique_sample_nodeTemporalCorrelation = zeros([unique_n_graphs, n_nodes]);
    unique_sample_temporalSmallWorldness = zeros([unique_n_graphs, 1]);
    unique_sample_burstiness = zeros([unique_n_graphs, 1]);

    now1 = tic();
%     for i = 1:unique_n_graphs
    parfor i = 1:unique_n_graphs
        now2 = tic();
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
        if size(strfind(data_name, 'auth'), 1) > 0 ...
            || size(strfind(data_name, 'protein100'), 1) > 0 ...
            ||  strcmp(data_name, 'wenbin_metro_flow') ...
        	||  strcmp(data_name, 'wenbin_scale_free_nodes_100')
            disp('do closeness centrality');
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
        run_time = toc(now2);
        disp([data_name ' iter ' mat2str(i) ' running time ' mat2str(run_time) ...
            ' estimated total time ' mat2str(run_time*unique_n_graphs)]);
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
    disp(['total running time ' mat2str(run_time)]);
    disp(['end ' data_name]);
    
    sample_betweenness = removeNAN(sample_betweenness);
    sample_closeness = removeNAN(sample_closeness);
    sample_broadcastCentrality = removeNAN(sample_broadcastCentrality);
    sample_receiveCentrality = removeNAN(sample_receiveCentrality);
    sample_temporalCorrelation = removeNAN(sample_temporalCorrelation);
    sample_nodeTemporalCorrelation = removeNAN(sample_nodeTemporalCorrelation);
    sample_temporalSmallWorldness = removeNAN(sample_temporalSmallWorldness);
    sample_temporalSmallWorldness = sample_temporalSmallWorldness(sample_temporalSmallWorldness > 0, :);
    disp(['sample_temporalSmallWorldness' mat2str(size(sample_temporalSmallWorldness))]);
    sample_burstiness = removeNAN(sample_burstiness);
    
    disp(['sample_betweenness ' mat2str(sum(sample_betweenness, [1, 2]))]);
    disp(['sample_closeness ' mat2str(sum(sample_closeness, [1, 2]))]);
    disp(['sample_broadcastCentrality ' mat2str(sum(sample_broadcastCentrality, [1, 2]))]);
    disp(['sample_receiveCentrality ' mat2str(sum(sample_receiveCentrality, [1, 2]))]);
    disp(['sample_temporalCorrelation ' mat2str(sum(sample_temporalCorrelation, [1, 2]))]);
    disp(['sample_nodeTemporalCorrelation ' mat2str(sum(sample_nodeTemporalCorrelation, [1, 2]))]);
    disp(['sample_temporalSmallWorldness ' mat2str(sum(sample_temporalSmallWorldness, [1, 2]))]);
    disp(['sample_burstiness ' mat2str(sum(sample_burstiness, [1, 2]))]);

end
