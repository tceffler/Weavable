function [data,sd,k,sk] = analyze(dirroot,d)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [data, sd, k, sk] = analyze(dirroot,d)
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
%  sd      : raw standard deviation of non-normalized, non-adjusted data.
%  k       : raw kurtosis value of non-normalized data, no 
%            adjustment for sample count
%  sk      : raw skewness value of non-normalized, non-adjusted data.
%
% example usage (assume data is ftq_X_counts.dat in the directory 
%                "/home/matt/data").
%
%   d = dir('/home/matt/data/*counts*.dat');
%   [data,sd,k,sk] = analyze('/home/matt/data',d);
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

% prepend single zero sample to all sample sequences
% data = [zeros(i,1) data];

% find maximum value that occurs in all data sets
maxval = max(data(:));
minval = min(data(:));

disp('Computing kurtosis and skewness values...');
for i=1:num_files
  xbar(i) = mean(data(i,:));
  sd(i) = std(data(i,:));
  kden(i) = npts*sd(i)^4;
  knum(i) = sum((data(i,:)-xbar(i)).^4);
  k(i) = kurtosis(data(i,:));
  sk(i) = skewness(data(i,:));
end
  disp('Min, Max='), disp(minval), disp(maxval);
  disp('Mean='), disp(xbar);
  disp('StdDev='), disp(sd);
  disp('Knum='), disp(knum);
  disp('Kden='), disp(kden);
  disp('kurtosis='), disp(k);
  disp('skewness='), disp(sk);
