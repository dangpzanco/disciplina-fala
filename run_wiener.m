clc; close all; clearvars

input_folder = 'data/speech+noise/';
output_folder = 'data/processed/wiener/';

input_files = dir([input_folder,'*.wav']);
num_files = length(input_files);

% Wiener
params.g  = 2;         % subtraction domain: 1=magnitude, 2=power [1]
params.e  = 2;         % gain exponent [1]

% params.of = 2;         % overlap factor = (fft length)/(frame increment) [2]
% params.ti = 16e-3;     % desired frame increment [0.016 seconds]
% params.ri = 0;         % set to 1 to round ti to the nearest power of 2 samples [0]
% params.am = 3;         % max oversubtraction factor [3]
% params.b  = 0.01;      % max noise attenutaion in power domain [0.01]
% params.al = -5;        % SNR for oversubtraction=am (set this to Inf for fixed a) [-5 dB]
% params.ah = 20;        % SNR for oversubtraction=1 [20 dB]
% params.ne = 0;         % noise estimation: 0=min statistics, 1=MMSE [0]
% params.bt = -1;        % threshold for binary gain or -1 for continuous gain [-1]
% params.mx = 0;         % input mixture gain [0]
% params.gh = 1;         % maximum gain for noise floor [1]
% params.rf = 0;         % round output signal to an exact number of frames [0]
% params.tf = 'g';       % selects time-frequency planes to output in the gg() variable ['g']
%                        % 'i' = input power spectrum
%                        % 'I' = input complex spectrum
%                        % 'n' = noise power spectrum
%                        % 'g' = gain
%                        % 'o' = output power spectrum
%                        % 'O' = output complex spectrum

% gain = max( 1 - a/SNR, min(1, b/SNR) )
% gain = 1 - a/SNR = (SNR - a)/SNR

for i=1:num_files
    
name = input_files(i).name;
name = name(1:end-4)

in_filename = [input_folder, input_files(i).name];
out_filename = [output_folder, input_files(i).name];

wiener_as(in_filename, out_filename)

% [noisy_speech, fs, nbits] = wavread(in_filename);
[noisy_speech, fs] = audioread(in_filename);

[enhanced_speech,gg,tt,ff,zo] = v_specsub(noisy_speech,fs,params);

% wavwrite(enhanced_speech, fs, nbits, out_filename);
audiowrite(out_filename, enhanced_speech, fs);

end


