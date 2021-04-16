function distance = mmd(X, Y, sigma)

% perform maximum mean discrepency
% K: the kernel matrix
% n_pos: the number of positive instances
% 
% n = size(K, 1);
% n_neg = n - n_pos;
% m = n_pos;
% n = n_neg;
% Kxx = K(1:n_pos, 1:n_pos);
% Kyy = K(n_pos + 1:n, n_pos+1:n);
% Kxy = K(1:n_pos, n_pos+1:n);
% 
% 
% %h_u = sum(Kxx(1:M,1:M)) - dgxx(1:M);
% %h_u = h_u + sum(Kyy(1:M,1:M)) - dgyy(1:M);
% %h_u = h_u - sum(Kxy(1:M,1:M)) - sum(Kxy(1:M,1:M)') + 2*dg;
% %mmd3 = sum(h_u)/M/(M-1);
% 
% sumKxx = sum(Kxx(:));
% sumKyy = sum(Kyy(:));
% sumKxy = sum(Kxy(:));
% mmd1 = sqrt(sumKxx/(m*m) +  sumKyy/(n * n) - 2/m/n * sumKxy);
% 
% %within_distance_pos = (sum(sum(K1)) - sum(diag(K1))) / (n_pos*(n_pos-1));
% %within_distance_neg = (sum(sum(K2)) - sum(diag(K2))) / (n_neg*(n_neg-1));
% %between_distance = sum(sum(K_cross)) / (n_pos * n_neg);
% 
% distance = mmd1; %within_distance_pos + within_distance_neg - 2 * between_distance;

Kxx = grbf(X, X, sigma);
Kxxnd = Kxx - diag(diag(Kxx), 0);
Kyy = grbf(Y, Y, sigma);
Kyynd = Kyy - diag(diag(Kyy), 0);
Kxy = grbf(X, Y, sigma);
m = size(Kxx, 1);
n = size(Kyy, 1);

u_xx = sum(Kxxnd, 'all') * (1 / (m * (m - 1)));
u_yy = sum(Kyynd, 'all') * (1 / (n * (n - 1)));
u_xy = sum(Kxy, 'all') / (m * n);

distance = u_xx + u_yy - 2 * u_xy;
