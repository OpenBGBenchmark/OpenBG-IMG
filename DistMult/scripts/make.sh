if [ ! -d "../OpenKE/openke/release" ]; then
PWD=$(pwd)
cd ../OpenKE/openke/
bash make.sh
cd $PWD
fi