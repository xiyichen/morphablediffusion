#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# username and password input
echo -e "\nIf you do not have an account you can register at https://flame.is.tue.mpg.de/ following the installation instruction."
read -p "Username (FLAME):" username
read -p "Password (FLAME):" password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading FLAME..."
mkdir -p assets/FLAME2020/
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O './FLAME2020.zip' --no-check-certificate --continue
unzip FLAME2020.zip -d assets/FLAME2020/
rm -rf FLAME2020.zip

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=TextureSpace.zip' -O './TextureSpace.zip' --no-check-certificate --continue
unzip TextureSpace.zip -d assets/FLAME2020/
rm -rf TextureSpace.zip

echo -e "\nDownloading MICA..."
mkdir -p assets/MICA/
wget -O assets/MICA/mica.tar "https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c0f96de/?dl=1"


echo -e "\nDownloading Morphable Diffusion..."
mkdir -p ckpt
cd ckpt
gdown 1lYm6todMDIJ8hahimzQ8v9STkda6dXxd
gdown 1KUjoDPKtNGM5UHABwsfMLVoMr7U8nnC7
cd ..