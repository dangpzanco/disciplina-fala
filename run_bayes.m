clc; close all; clearvars

input_folder = '../data/speech+noise/';
output_folder = '../data/processed/bayes/';

input_files = dir([input_folder,'*.wav']);
num_files = length(input_files);

% Bayes
params.lg = 2   % MMSE target: 0=amplitude, 1=log amplitude, 2=perceptual Bayes [1]

% params.of = 2            % overlap factor = (fft length)/(frame increment) [2]
% params.ti = 16e-3        % desired frame increment [0.016 seconds]
% params.ri = 0            % set to 1 to round ti to the nearest power of 2 samples [0]
% params.ta = 0.396        % time const for smoothing SNR estimate [0.396 seconds]
% params.gx = 1000         % maximum posterior SNR as a power ratio [1000 = +30dB]
% params.gn = 1            % min posterior SNR as a power ratio when estimating prior SNR [1 = 0dB]
% params.gz = 0.001        % min posterior SNR as a power ratio [0.001 = -30dB]
% params.xn = 0            % minimum prior SNR [0]
% params.xb = 1            % bias compensation factor for prior SNR [1]
% params.tn = 0.5          % smoothing time constant for noise estimation [0.5 s]
% params.le = 0.15         % VAD threshold: log(p/(1-p)) where p is speech prob in a freq bin; use -Inf to prevent updating [0.15 (=>p=0.54)]
% params.tx = 0.06         % initial noise interval [0.06 s]
% params.ne = 0            % noise estimation: 0=min statistics, 1=MMSE [0]
% params.bt = -1           % threshold for binary gain or -1 for continuous gain [-1]
% params.mx = 0            % input mixture gain [0]
% params.rf = 0            % round output signal to an exact number of frames [0]
% params.tf = 'g'          % selects time-frequency planes to output in the gg() variable ['g']
%                 % 'i' = input power spectrum
%                 % 'I' = input complex spectrum
%                 % 'n' = noise power spectrum
%                 % 'z' = "posterior" SNR (i.e. (S+N)/N )
%                 % 'x' = "prior" SNR (i.e. S/N )
%                 % 'g' = gain
%                 % 'o' = output power spectrum
%                 % 'O' = output complex spectrum



for i=1:num_files
    
name = input_files(i).name;
name = name(1:end-4)

in_filename = [input_folder, input_files(i).name];
out_filename = [output_folder, input_files(i).name];

% stsa_wlr(in_filename, out_filename)

[noisy_speech, fs, nbits] = wavread(in_filename);

[enhanced_speech,gg,tt,ff,zo] = v_ssubmmsev(noisy_speech,fs,params);

wavwrite(enhanced_speech, fs, nbits, out_filename);


end


