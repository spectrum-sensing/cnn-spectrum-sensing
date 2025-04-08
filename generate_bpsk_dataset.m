clc;
clear;
close all;

% Setup
n = 1024;
fs = 1000;
fc = 100;
t = (0:n-1)/fs;
mkdir('signal_dataset/occupied');
mkdir('signal_dataset/vacant');

for i = 1:500
    %% -------- OCCUPIED SIGNAL --------
    data = randi([0 1], 1, n);
    bpsk = 2*data - 1;
    carrier = cos(2*pi*fc*t);
    bpsk_signal = bpsk .* carrier;

    snr_db = -20 + (40)*rand();  % Random SNR from -20 to +20 dB
    noisy_signal = awgn(bpsk_signal, snr_db, 'measured');

    % Create figure with visible SNR
    f1 = figure('Visible','off');
    plot(t, noisy_signal, 'k', 'LineWidth', 1);
    title(sprintf('Occupied Signal at SNR = %.1f dB', snr_db), 'FontWeight','bold');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    filename = sprintf('signal_dataset/occupied/occupied_%03d_SNR_%+.1f.png', i, snr_db);
    saveas(f1, filename);
    close(f1);

    %% -------- VACANT SIGNAL --------
    noise_only = randn(1, n);

    f2 = figure('Visible','off');
    plot(t, noise_only, 'k', 'LineWidth', 1);
    title('Vacant Signal (Noise Only)', 'FontWeight','bold');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    filename2 = sprintf('signal_dataset/vacant/vacant_%03d.png', i);
    saveas(f2, filename2);
    close(f2);
end
