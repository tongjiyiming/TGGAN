graphs = importdata('auth_user_0.txt');
n_times = 4;
nNodes = 27;

d_col = 1;
i_col = 2;
j_col = 3;
t_col = 4;
directed = 1;
time_interval = 1. / n_times;
ind = graphs(:, t_col) < 1;
graphs = graphs(ind, :);
graphs(:, t_col) = ceil(graphs(:, t_col) / time_interval);
graphs(:, i_col) = graphs(:, i_col) + 1;
graphs(:, j_col) = graphs(:, j_col) + 1;
graphs(:, d_col) = graphs(:, d_col) - min(graphs(:, d_col)) + 1;
unique_d = unique(graphs(:, d_col));

d_size = size(unique_d, 1);

ind = graphs(:, d_col) == unique_d(1);
contactSequence = graphs(ind, :);
contactSequence = contactSequence(:, 2:4);
timeInterval = [0 0.5];
plotDNarc( contactSequence, nNodes, ...
    timeInterval)