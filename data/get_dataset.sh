SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/
mkdir -p shapenetcore_part

wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip -d shapenetcore_part/
rm shapenetcore_partanno_segmentation_benchmark_v0.zip

cd shapenetcore_part
mv shapenetcore_partanno_segmentation_benchmark_v0/* ./
rm -r shapenetcore_partanno_segmentation_benchmark_v0/
cd $SCRIPTPATH
