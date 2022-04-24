if [ ! -d "release" ]; then
mkdir release
fi
g++ ./base/Base.cpp -fPIC -shared -o ./release/Base.so -pthread -O3 -march=native
