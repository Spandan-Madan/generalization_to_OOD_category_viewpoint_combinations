echo 'Building directory structure...'
cd ../
mkdir -p data/
cd data/
mkdir -p biased_cars
cd biased_cars

echo 'Directories done. Downloading data...'
wget https://dataverse.harvard.edu/api/access/datafile/5793795
mv 5793795 att_dict_simplified.p

wget https://dataverse.harvard.edu/api/access/datafile/5347218
mv 5347218 biased_cars_1.zip

wget https://dataverse.harvard.edu/api/access/datafile/5347225
mv 5347225 biased_cars_2.zip

wget https://dataverse.harvard.edu/api/access/datafile/5347224
mv 5347224 biased_cars_3.zip

wget https://dataverse.harvard.edu/api/access/datafile/5347223
mv 5347223 biased_cars_4.zip

echo 'Downloaded, unzipping files...'
unzip biased_cars_1.zip
unzip biased_cars_2.zip
unzip biased_cars_3.zip
unzip biased_cars_4.zip

echo 'moving files to correct directories...'
mv biased_cars_1/* ./ 
mv biased_cars_2/* ./
mv biased_cars_3/* ./
mv biased_cars_4/* ./

echo 'Removing unwanted files...'

rm -r biased_cars_1
rm -r biased_cars_2
rm -r biased_cars_3
rm -r biased_cars_4
rm biased_cars_1.zip
rm biased_cars_2.zip
rm biased_cars_3.zip
rm biased_cars_4.zip

rm -r __MACOSX

echo 'All done!'
