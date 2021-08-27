for f in `find . -name "out.cpu.*"`
do 
echo -n checking $f ...
grep Grind $f | awk '{printf(" FOM = %.1lf\n",1.0/$4)}'
done
