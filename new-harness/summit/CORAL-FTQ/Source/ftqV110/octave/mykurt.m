#
#
#
1;

myFiles = glob("*counts.dat");
printf("\n");

for i = 1:rows(myFiles)
     X = load(myFiles{i});
     # add a zero to the array to normalize the values for Kurtosis
     # intends to neutralize differences in processor speeds
     X = [0; X];
     X = double(X) ./ double(max(X(:)));
     k = kurtosis(X) /length(X); 
# to get normalized number?

     s = skewness(X);
     printf("%s\t%f\t%f\n", myFiles{i}, k, s);
endfor


