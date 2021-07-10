#dnn=resnet34
#batch_size=64
dnn=inceptionv4
batch_size=16
nworkers="${nworkers:-64}"
rdma="${rdma:-1}"
#kfac_name=inverse_opt
kfac_name="${kfac_name:-inverse_opt}"
#epochs=1 kfac_name=$kfac_name kfac=0 dnn=$dnn nworkers=1 rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 kfac_name=$kfac_name kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh

dnn=densenet201
batch_size=16
#epochs=1 kfac_name=$kfac_name kfac=0 dnn=$dnn nworkers=1 rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 kfac_name=$kfac_name kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh

dnn=resnet50
batch_size=32
#epochs=1 kfac_name=$kfac_name kfac=0 dnn=$dnn nworkers=1 rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 kfac_name=$kfac_name kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh

dnn=resnet152
batch_size=8
#epochs=1 kfac_name=$kfac_name kfac=0 dnn=$dnn nworkers=1 rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 kfac_name=$kfac_name kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
epochs=1 kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
#epochs=1 exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./horovod_mpi_cj.sh
