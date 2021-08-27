function [data,xb,sd,k] = wanalyze(dirroot,d)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [data, xb, sd, k] = wanalyze(dirroot,d)
% 
% arguments:
%  dirroot : root directory for the data files to read in
%  d       : the dir() output containing the sequence of files to 
%            read
%
% returns:
%  data    : given m files to read in with n entries per file, an 
%            mxn matrix containing one data set per row.  data
%            rows are ordered lexicographical by filename, NOT
%            numerically.  Beware!
%  xb      : mean of scaled noise of the data
%  sd      : standard deviation of scaled noise of the data 
%  k       : kurtosis value of scaled noise of the data
%
% example usage (assume data is ftq_X_counts.dat in the directory 
%                "/home/matt/data").
%
%   d = dir('/home/matt/data/*counts*.dat');
%   [data,xb,sd,k] = analyze('/home/matt/data',d);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_files = length(d);

disp('Loading data...');

% iterate through files.  NOTE: ordering is lexicographic, NOT
% numerical!
for i=1:num_files
  disp(d(i).name);
  data(i,:) = load(strcat([dirroot '/' d(i).name]));
end

% count number of samples.  all sample sequences assumed to be
% the same length.
npts = length(data(1,:));

% find maximum value that occurs in all data sets
maxval = max(data(:));
minval = min(data(:));

% Remove the work (minval time) to expose the noise and scale the noise
% as a percentage of the work.
disp('Computing scaled noise...');
for i=1:num_files
  data(i,:) = (data(i,:).-minval)./minval;
end

disp('Computing Mean, Standard Deviation and Kurtosis...');
for i=1:num_files
  xb(i) = mean(data(i,:));
  sd(i) = std(data(i,:));
  knum(i) = sum((data(i,:)-xb(i)).^4);
  kden(i) = npts*sd(i)^4;
  k(i) = kurtosis(data(i,:));
end
  disp('Min, Max='), disp(minval), disp(maxval);
  disp('Mean='), disp(xb);
  disp('StdDev='), disp(sd);
  disp('Knum='), disp(knum);
  disp('Kden='), disp(kden);
  disp('kurtosis='), disp(k);
