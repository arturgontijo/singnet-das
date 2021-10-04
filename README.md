# singnet-das


```
git clone https://github.com/arturgontijo/singnet-das.git
cd singnet-das
docker build . -t opencog-singnet-das
docker run --rm --name SINGNET_DAS -v singnet-das:/opt/singnet/projects/singnet-das/ -p8888:8888 -it opencog-singnet-das
```
