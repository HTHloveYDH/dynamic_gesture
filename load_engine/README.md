# Build (compile)

```
cd load_engine
mkdir build
cd build
cmake ..
make -j$(nproc) 
```

# Run
```
cd load_engine/build
./dynamic_gesture --datadir ../model
```