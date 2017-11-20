function out = save_mats(input)
%% function for saving SPM-formatted MMN_longitudinal data as simple mat-structs

% author = Andreas
% date = 2017-10-05

out = [];

addpath('/usr/local/common/matlab_toolbox/spm12_latest');
spm('defaults', 'eeg');

%% initial vars

Pdir = '/projects/MINDLAB2012_16-EEG-Phoneme-Longitudinal-MMN/scratch/';
extName = 'ica';
saveExt = 'for_python';
if ~exist(fullfile(Pdir,saveExt),'dir')
    mkdir(fullfile(Pdir,saveExt))
end
times = {'longi-MMN-T0','longi-MMN-T1','longi-MMN-T2','longi-MMN-T3'};
t = times{input(2)};
conds = {'ha','sh'};
prefix = 'arespm';
subjs = {'arab1','arab2','arab3','arab4','arab5','arab6','arab7','arab8', ...
    'dari1','dari2','dari3','dari4','dari5','dari6','dari7','dari8','dari9','dari10','dari11','dari12'};
subjid = subjs{input(1)};
stims = [0 28 49 70;
    1 4 7 11]';

%% load, split up, and save as individual conditions

D = spm_eeg_load(fullfile(Pdir,t,extName,sprintf('%s_%s_%s_ica.mat',prefix,t(end-1:end),subjid)));
condList = conditions(D);
condNames = unique(condList);
for i = 1:length(condNames)
    a = D(:,:,find(strcmp(condNames{i},condList))); % pick out data for the individual conditions 
    cond = find(strcmp(condNames{i}(5:6),conds)); % find index for either arabic or dari stimuli
    stim = find(stims(:,cond)==str2double(condNames{i}(end-1:end)))-1; % converting the stim "number" to an index from 0-4 (so that the file name becomes *_std0:3 and *_dev1:3 
    save(fullfile(Pdir,saveExt,sprintf('S%d_C%d_T%d_%s%d.mat',input(1),cond,str2double(t(end)),condNames{i}(1:3),stim)),'a');
    clear a cond stim
end


