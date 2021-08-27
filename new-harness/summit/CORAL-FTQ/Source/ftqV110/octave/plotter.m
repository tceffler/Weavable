function plotter(data,xb,sd,k,rawstride,nbins)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotter for output of analyze.m
%
% input:
%  data : data array returned by analyze.m with one FTQ vector 
%         per row.
%  xb   : Mean computed by analyze.m
%  sd   : Standard deviation computed by analyze.m
%         used to title graphs.
%  k    : kurtosis values computed by analyze.m.
%         used to title graphs.
%  rawstride : instead of plotting ALL data points from the
%              raw data, plot every rawstride-th element.  this
%              is simply to prevent gnuplot from choking on
%              huge data sets.  a value of 100-1000 is good.
%  nbins : number of bins for computing histograms.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % get data dimensions
  [numdata,numelt] = size(data);
  
  % compute subplot layout parameters
  spR = ceil(sqrt(numdata));
  spC = ceil(numdata/ceil(sqrt(numdata)));
  
  % determine maximum data value
  maxval = max(data(:));
  
  % plot raw data
  figure;
  for i=1:numdata
    subplot(spR,spC,i);
    plot(data(i,1:rawstride:end));
    title(sprintf('RAW:xb=%f sd=%f k=%f',xb(i),sd(i),k(i)));
    ylim([0 maxval]);
    xlim([0 ceil(numelt/rawstride)]);
  end
  
  % plot log histograms
  figure;
  for i=1:numdata
    subplot(spR,spC,i);
# Make sure the min of the histogram bins is zero work....
    h=hist([0 data(i,:)], nbins);
    plot(log(h+1));
    title(sprintf('LOG-HIST:xb=%f sd=%f k=%f',xb(i),sd(i),k(i)))
  end
