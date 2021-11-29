clear; clc; close all;

M = 4;          % Modulation order
Nf = 512;       % Length of FFT/IFFT
Ncp = Nf/4;     % Length of cyclic prefix
Nd = Nf/2 - 1;  % Length of data subcarriers
Ns = 1e5;       % Total symbols

qammod_in = randi([0, M-1], Ns, Nd);

% QAM Mapping
qammod_out = qammod(qammod_in, M);
dc = zeros(Ns, 1);
% dc = zeros(Ns, 1)+12;
% Insert zeros and Hermitian symmetry
subcarriers = [dc, qammod_out, zeros(Ns, 1), conj(qammod_out(:, end:-1:1))];

% Modulation
txSig_ifft = ifft(subcarriers, Nf, 2);

% Insert cyclic prefix
txSig_cp = [txSig_ifft(:, Nf-Ncp+1:Nf), txSig_ifft];

% Normalization
nc = max(max(abs(txSig_cp)));
txSig = txSig_cp ./ nc;

% CCDF
[cdf, papr] = ecdf(calculatePAPR(txSig));

LineWidth = 1.15;
FontSize = 12;
MarkerSize = 14;
h1 = figure;
set(gcf,'color','w');
semilogy(papr, 1-cdf, 'LineWidth', LineWidth);
grid on

function [ papr ] = calculatePAPR( sig )

papr = 10 * log10(max(abs(sig.^2),[],2)./mean(abs(sig.^2),2));
P = abs(sig.^2);
end
