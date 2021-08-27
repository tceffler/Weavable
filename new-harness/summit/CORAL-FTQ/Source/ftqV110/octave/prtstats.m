function [dt] = prtstats(dirroot,fname)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function fixup(dirroot,fname)
% 
% arguments:
%  dirroot : root directory for the data files to read in
%  d       : the dir() output containing the sequence of files to 
%            read
%
% returns:
%
% example usage (assume data is ftq_X_counts.dat in the directory 
%                "/home/matt/data").
%
%   d = dir('/home/matt/data/*counts*.dat');
%   prtstats('/home/matt/data',d);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


disp('Loading data...');

% iterate through files.  NOTE: ordering is lexicographic, NOT
% numerical!
disp("loading file "), disp(fname);
t = load(fname);

% count number of samples.
n = length(t);
dt = t(1:n-1);

% take the delta of successive elements
disp('Computing deltas');
for i=1:n-1
  dt(i) = t(i+1)-dt(i);
end

% find maximum value that occurs in all data sets
disp('Computing statisics');
maxval = max(dt);
minval = min(dt);
m=mean(dt);
sd=std(dt);
k=kurtosis(dt);

disp('Min, Max = '), disp(minval), disp(maxval);
disp('Mean     = '), disp(m);
disp('StdDev   = '), disp(sd);
disp('Kurtosis = '), disp(k);

