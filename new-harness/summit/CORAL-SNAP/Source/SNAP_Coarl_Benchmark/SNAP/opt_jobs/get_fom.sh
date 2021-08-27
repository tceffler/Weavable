for f in `find . -name "out.gpu.*"`
do 
echo -n checking $f ...
grep Grind $f | awk '{printf(" FOM = %.1lf\n",1.0/$4)}'
done
