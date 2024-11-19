root_dir=data
# data_dirname_list=(Clothing_Shoes_and_Jewelry)
# data_urlname_list=(Clothing_Shoes_and_Jewelry)

# data_dirname_list=(Grocery CDs Luxury_Beauty Tools)
# data_urlname_list=(Grocery_and_Gourmet_Food CDs_and_Vinyl Luxury_Beauty Tools_and_Home_Improvement)

data_dirname_list=(Electronics)
data_urlname_list=(Electronics)

# https://jmcauley.ucsd.edu/data/amazon_v2/index.html

len=${#data_dirname_list[@]}
for (( i=0; i<$len; i++ )); do
    mkdir -p $root_dir/${data_dirname_list[$i]}
    wget https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/${data_urlname_list[$i]}_5.json.gz -O $root_dir/${data_dirname_list[$i]}/5-core.json.gz --no-check-certificate
    gzip -d $root_dir/${data_dirname_list[$i]}/5-core.json.gz
    wget https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_${data_urlname_list[$i]}.json.gz -O $root_dir/${data_dirname_list[$i]}/metadata.json.gz --no-check-certificate
    gzip -d $root_dir/${data_dirname_list[$i]}/metadata.json.gz
done
