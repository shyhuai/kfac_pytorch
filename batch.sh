#dnn=resnet34
#batch_size=64
dnn=inceptionv4
batch_size=16
nworkers=64
#epochs=1 kfac_name=inverse kfac=0 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

dnn=densenet201
batch_size=16
nworkers=64
epochs=1 kfac_name=inverse kfac=0 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh



#dnn=resnet152
#batch_size=8
#epochs=1 kfac_name=inverse kfac=0 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 kfac_name=inverse kfac=0 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

#epochs=1 kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh


#dnn=resnet50
#batch_size=32
#epochs=1 kfac_name=inverse kfac=0 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 kfac_name=inverse kfac=0 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor kfac_name=inverse kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

#epochs=1 kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=inverse_opt kfac=1 dnn=$dnn nworkers=16 rdma=1 batch_size=$batch_size ./horovod_mpi_cj.sh

