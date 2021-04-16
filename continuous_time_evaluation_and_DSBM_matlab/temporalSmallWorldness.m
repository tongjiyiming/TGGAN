function [ smallWorldness ] = temporalSmallWorldness(contactSequence,...
    nNodes,nTimes,C,L_mat,directed)
%% modify by TGGAN
% temporalSmallWorldness returns values needed to compute the temporal
% small worldness of a dynamic network.
%
% Inputs:
%       contactSequence = nEdges x 3 matrix encoding contacts between node
%           i,j at time t by (i,j,t). 
%       directed = 1 if the dynamic network is directed, 0 otherwise.
%
% Optional Inputs:
%       nNodes = number of nodes in the dynamic network. Default is all
%           nodes which appear in contactSequence (have at least one
%           contact).
%
% Outputs:
%       C = temporal correlation of input dynamic network
%       L = efficientcy of dynamic network
%
%
%
% Reference: Ann E. Sizemore and Danielle S. Bassett, "Dynamic Graph 
% Metrics: Tutorial, Toolbox, and Tale." Submitted. (2017)
%
% Main function:


if ~exist('nNodes','var') || isempty(nNodes)
    nNodes = length(unique([contactSequence(:,1); contactSequence(:,2)]));
end

if ~exist('C','var') || isempty(nNodes)
    [C] = temporalCorrelation( contactSequence,nNodes,nTimes,directed);
end

% compute efficiency
if ~exist('L_mat','var') || isempty(nNodes)
    [~, ~, L_mat] = betweennessCentrality(contactSequence,nNodes,nTimes,directed);
end
L = (1/(nNodes*(nNodes-1)))*sum(L_mat(:));

% re_sequence = randomizedEdges(contactSequence, 100);
re_sequence = randomPermutedTimes(contactSequence);

[C_re] = temporalCorrelation( re_sequence,nNodes,nTimes,directed);
[~, ~, L_mat_re] = betweennessCentrality(re_sequence,nNodes,nTimes,directed);
L_re = (1/(nNodes*(nNodes-1)))*sum(L_mat_re(:));

smallWorldness = (C / C_re) / (L / L_re);
if isnan(smallWorldness) || isinf(smallWorldness)
    smallWorldness = 0;
end

end

