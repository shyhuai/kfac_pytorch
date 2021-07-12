dnn=resnet56
batch_sizes=( "256" "512" "1024" ) #"2048" "4096" )
nworkers=1
lrs=( "0.001" "0.01" "0.1" "1.0" )
#sparse_ratios=( "1" "0.01" ) #"0.001" ) #"0.001" )
sparse_ratios=( "0.1" ) #"0.001" )
for lr in "${lrs[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        for sr in "${sparse_ratios[@]}"
        do
            if [ "$sr" = "1" ]; then
                #gpuids=1 rdma=0 batch_size=$batch_size lr=$lr nworkers=$nworkers dnn=$dnn kfac_name=eigen sparse_ratio=$sr kfac=1 ./horovod_mpi_cj.sh &
                gpuids=0 rdma=0 batch_size=$batch_size lr=$lr nworkers=$nworkers dnn=$dnn kfac_name=inverse_naive sparse_ratio=$sr kfac=1 ./horovod_mpi_cj.sh & 
                gpuids=1 rdma=0 batch_size=$batch_size lr=$lr nworkers=$nworkers dnn=$dnn kfac_name=inverse_naive sparse_ratio=$sr kfac=0 ./horovod_mpi_cj.sh &
            else
                #damping=0.001 rdma=0 batch_size=$batch_size lr=$lr nworkers=$nworkers dnn=$dnn kfac_name=sparse_hessian sparse_ratio=$sr kfac=1 ./horovod_mpi_cj.sh
                gpuids=2 rdma=0 batch_size=$batch_size lr=$lr nworkers=$nworkers dnn=$dnn kfac_name=sparse_sgd sparse_ratio=$sr kfac=1 ./horovod_mpi_cj.sh &
                gpuids=3 rdma=0 batch_size=$batch_size lr=$lr nworkers=$nworkers dnn=$dnn kfac_name=minibatch_fisher sparse_ratio=$sr kfac=1 ./horovod_mpi_cj.sh &
                #gpuids=0 rdma=0 batch_size=$batch_size lr=$lr nworkers=$nworkers dnn=$dnn kfac_name=sparse_sgd sparse_ratio=0.001 kfac=1 ./horovod_mpi_cj.sh &
                #gpuids=1 rdma=0 batch_size=$batch_size lr=$lr nworkers=$nworkers dnn=$dnn kfac_name=minibatch_fisher sparse_ratio=0.001 kfac=1 ./horovod_mpi_cj.sh &
                #gpuids=2 rdma=0 batch_size=$batch_size lr=$lr nworkers=$nworkers dnn=$dnn kfac_name=sparse_sgd sparse_ratio=0.01 kfac=1 ./horovod_mpi_cj.sh &
                #gpuids=3 rdma=0 batch_size=$batch_size lr=$lr nworkers=$nworkers dnn=$dnn kfac_name=minibatch_fisher sparse_ratio=0.01 kfac=1 ./horovod_mpi_cj.sh &
            fi
        done
        wait
    done
done
