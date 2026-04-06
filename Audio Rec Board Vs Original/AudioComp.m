% =========================================================
% Fan Fault Audio Comparison
% Original (played from PC) vs Recorded (captured by board)
% =========================================================

clear; clc; close all;

% --- Load Files ---
[original, fs_o] = audioread('fan_original.wav');
[recorded, fs_r] = audioread('fan_recorded.wav');


% --- Average all channels to mono ---
original = mean(original, 2);
recorded = mean(recorded, 2);

% --- Resample to match sample rates if different ---
if fs_o ~= fs_r
    recorded = resample(recorded, fs_o, fs_r);
    fs_r = fs_o;
end
fs = fs_o;

% --- Normalize both to [-1, 1] ---
original = original / max(abs(original));
recorded = recorded / max(abs(recorded));

% --- Trim to same length ---
min_len = min(length(original), length(recorded));
original = original(1:min_len);
recorded = recorded(1:min_len);

% --- Time and frequency axes ---
N = min_len;
t = (0:N-1) / fs;
f = (0:N/2-1) * fs / N;

% =========================================================
% FFT of both
% =========================================================
fft_orig = abs(fft(original)) / N;
fft_orig = fft_orig(1:N/2) * 2;

fft_rec = abs(fft(recorded)) / N;
fft_rec = fft_rec(1:N/2) * 2;

% =========================================================
% METRICS
% =========================================================

% 1. RMS Energy
rms_orig = sqrt(mean(original.^2));
rms_rec  = sqrt(mean(recorded.^2));
fprintf('--- RMS Energy ---\n');
fprintf('Original:  %.4f\n', rms_orig);
fprintf('Recorded:  %.4f\n', rms_rec);

% 2. Peak frequency
[~, idx_o] = max(fft_orig);
[~, idx_r] = max(fft_rec);
fprintf('\n--- Dominant Frequency ---\n');
fprintf('Original:  %.2f Hz\n', f(idx_o));
fprintf('Recorded:  %.2f Hz\n', f(idx_r));

% 3. Spectral Centroid
centroid_orig = sum(f .* fft_orig') / sum(fft_orig);
centroid_rec  = sum(f .* fft_rec')  / sum(fft_rec);
fprintf('\n--- Spectral Centroid ---\n');
fprintf('Original:  %.2f Hz\n', centroid_orig);
fprintf('Recorded:  %.2f Hz\n', centroid_rec);

% 4. Crest Factor
crest_orig = max(abs(original)) / rms_orig;
crest_rec  = max(abs(recorded)) / rms_rec;
fprintf('\n--- Crest Factor ---\n');
fprintf('Original:  %.4f\n', crest_orig);
fprintf('Recorded:  %.4f\n', crest_rec);

% 5. Spectral Difference Score (MSE between FFT magnitudes)
spectral_mse = mean((fft_orig - fft_rec).^2);
fprintf('\n--- Spectral MSE (0 = identical) ---\n');
fprintf('Score: %.8f\n', spectral_mse);


% 7. Cross-correlation (checks time alignment / delay)
[xc, lags] = xcorr(original, recorded, 'normalized');
[~, max_idx] = max(xc);
delay_samples = lags(max_idx);
delay_ms = delay_samples / fs * 1000;




% 8. Align signals and recompute correlation
if delay_samples > 0
    recorded_aligned = recorded(delay_samples+1:end);
    original_aligned = original(1:length(recorded_aligned));
elseif delay_samples < 0
    original_aligned = original(abs(delay_samples)+1:end);
    recorded_aligned = recorded(1:length(original_aligned));
else
    original_aligned = original;
    recorded_aligned = recorded;
end



% 9. Frequency domain correlation (more reliable for noise signals)
fft_orig_aligned = abs(fft(original_aligned)) / min_len;
fft_rec_aligned  = abs(fft(recorded_aligned)) / min_len;
fft_orig_aligned = fft_orig_aligned(1:floor(min_len/2)) * 2;
fft_rec_aligned  = fft_rec_aligned(1:floor(min_len/2))  * 2;

r_spectral = corrcoef(fft_orig_aligned, fft_rec_aligned);
fprintf('\n--- Spectral Correlation (1 = identical) ---\n');
fprintf('Correlation: %.4f\n', r_spectral(1,2));

% =========================================================
% PLOTS                                                <-- THEN PLOTS AS BEFORE
% =========================================================

% =========================================================
% PLOTS
% =========================================================

figure('Name', 'Original vs Recorded Comparison', 'NumberTitle', 'off');

% Time domain overlay
subplot(3,1,1);
plot(t, original, 'b', 'DisplayName', 'Original'); hold on;
plot(t, recorded, 'r', 'DisplayName', 'Recorded');
xlabel('Time (s)'); ylabel('Amplitude');
title('Time Domain'); legend; grid on;

% FFT overlay
subplot(3,1,2);
plot(f/1000, fft_orig, 'b', 'DisplayName', 'Original'); hold on;
plot(f/1000, fft_rec,  'r', 'DisplayName', 'Recorded');
xlabel('Frequency (kHz)'); ylabel('Magnitude');
title('FFT Spectrum Comparison');
xlim([0 10]); legend; grid on;

% PSD overlay
subplot(3,1,3);
[psd_orig, f_psd] = pwelch(original, 1024, 512, 1024, fs);
[psd_rec,  ~    ] = pwelch(recorded, 1024, 512, 1024, fs);
semilogy(f_psd/1000, psd_orig, 'b', 'DisplayName', 'Original'); hold on;
semilogy(f_psd/1000, psd_rec,  'r', 'DisplayName', 'Recorded');
xlabel('Frequency (kHz)'); ylabel('Power/Hz');
title('PSD Comparison'); legend; grid on;

% Cross-correlation plot
figure('Name', 'Cross-Correlation', 'NumberTitle', 'off');
plot(lags/fs*1000, xc);
xlabel('Lag (ms)'); ylabel('Normalized Correlation');
title(sprintf('Cross-Correlation (peak delay = %.2f ms)', delay_ms));
grid on;