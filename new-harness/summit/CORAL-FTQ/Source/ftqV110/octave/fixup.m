function fixup(dirroot,fname)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function fixup(dirroot,fname)
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
%  k       : raw kurtosis value on non-normalized data, no 
%            adjustment for sample count
%  s       : raw skewness value on non-normalized, non-adjusted data.
%  normk   : kurtosis computed on normalized data (data(:) / maxval) 
%            and divided by the number of samples.
%  norms   : skewness computed on normalized/adjusted data.
%
%
% example usage (assume data is ftq_X_counts.dat in the directory 
%                "/home/matt/data").
%
%   d = dir('/home/matt/data/*counts*.dat');
%   [data,normk,k] = analyze('/home/matt/data',d,100,0,1);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


disp('Loading data...');

% iterate through files.  NOTE: ordering is lexicographic, NOT
% numerical!
disp("loading file "), disp(fname);
data(:) = load(fname);

% count number of samples.  all sample sequences assumed to be
% the same length.
npts = length(data);


% find maximum value that occurs in all data sets
maxval = max(data);

disp('Computing fixed vals');
for i=1:npts
  if( (maxval-data(i))/maxval > 0.01) 
    data(i) = maxval;
  endif
end
  save -text fixed_counts.dat data;

