clc; close all; clearvars


data_folder = '../data/';

%  ['filename', 'speech_name', 'noise_name', 'realization', 'SNR']

% meta_filename = [data_folder, 'metadata.csv'];
meta_filename = 'speechmetrics_results.csv';

opts = detectImportOptions(meta_filename);
metadata = readtable(meta_filename);
num_files = size(metadata, 1);

technique_list = {'noisy', 'wiener', 'bayes', 'binary'};
num_techniques = length(technique_list);

llr_score = zeros(num_files,1);
csii_score = zeros(num_files,1);
for i=1:num_files

technique = metadata.technique{i};


if strcmp(technique, 'noisy')
    processed_folder = [data_folder, 'speech+noise/']; 
else
    processed_folder = [data_folder, 'processed/', technique, '/'];
end

speech_filename = [processed_folder, metadata.speech_name{i}, '_SNOW0_inf.wav'];
processed_filename = [processed_folder, metadata.filename{i}, '.wav'];

llr_result = comp_llr(speech_filename, processed_filename);
[csii_high, csii_mid, csii_low] = CSII(speech_filename, processed_filename);

llr_score(i) = llr_result;
csii_score(i) = csii_mid;


disp([technique, ': ', metadata.filename{i}, ' ', num2str(llr_result), ' ', num2str(csii_mid)])

end

metadata.llr = llr_score;
metadata.csii = csii_score;

writetable(metadata,'matlab_results.csv')

% 
%     # for i in trange(num_files):
%     for i in tqdm.tqdm(file_ind):
%     # for i in range(num_files):
%         
%         # Get filename
%         speech_filename = processed_folder / f"{metadata['speech_name'][i]}_SNOW0_inf.wav"
%         processed_filename = processed_folder / f"{metadata['filename'][i]}.wav"
% 
%         scores = {}
%         scores['llr'] = utils.llr(processed_filename, speech_filename)
%         scores['siib'] = utils.siib(processed_filename, speech_filename)
% 
%         print(scores)
% 
%         exit()
% 
%         filename.append(metadata['filename'][i])
%         speech_name.append(metadata['speech_name'][i])
%         noise_name.append(metadata['noise_name'][i])
%         realization.append(metadata['realization'][i])
%         SNR.append(metadata['SNR'][i])
%         technique.append(technique_list[k])
%         llr_score.append(scores['llr'])
%         llr_score.append(scores['siib'])