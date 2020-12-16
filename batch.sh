epochs=1 kfac_name=inverse kfac=1 dnn=resnet34 nworkers=64 rdma=1 batch_size=64 ./horovod_mpi_cj.sh
epochs=1 kfac_name=inverse kfac=1 dnn=resnet34 nworkers=16 rdma=1 batch_size=64 ./horovod_mpi_cj.sh
epochs=1 kfac_name=inverse_opt kfac=1 dnn=resnet34 nworkers=64 rdma=1 batch_size=64 ./horovod_mpi_cj.sh
epochs=1 kfac_name=inverse_opt kfac=1 dnn=resnet34 nworkers=16 rdma=1 batch_size=64 ./horovod_mpi_cj.sh
