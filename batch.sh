dnn=resnet152
batch_size=16
epochs=1 kfac_name=inverse kfac=0 dnn=$dnn nworkers=64 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 kfac_name=inverse kfac=0 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

epochs=1 kfac_name=inverse kfac=1 dnn=$dnn nworkers=64 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=64 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=64 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=64 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=64 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

epochs=1 kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=64 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=64 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=64 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=64 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

