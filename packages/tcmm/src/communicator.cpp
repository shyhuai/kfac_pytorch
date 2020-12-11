#include <torch/extension.h>
#include "communicator.h"


Communicator::Communicator(int rank, int size):m_rank(rank), m_size(size) {
	if (rank == 0) ncclGetUniqueId(&m_nccl_id);
	CUDACHECK(cudaStreamCreate(&m_stream));
	MPICHECK(MPI_Bcast((void *)&m_nccl_id, sizeof(m_nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
	NCCLCHECK(ncclCommInitRank(&m_nccl_comm, size, m_nccl_id, rank));
}

Communicator::~Communicator() {
	ncclCommDestroy(m_nccl_comm);
}

void Communicator::synchronize() {
    CUDACHECK(cudaStreamSynchronize(m_stream));
}

//void Communicator::allReduce(const void* sendbuff, void* recvbuff, int size) {
//    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, m_nccl_comm, m_stream));
//}

void Communicator::allReduce(torch::Tensor tensor) {
    NCCLCHECK(ncclAllReduce(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), ncclFloat, ncclSum, m_nccl_comm, m_stream));
}

//void Communicator::multiBcast(vector<torch::Tensor> &tensor_list, void (*op)(torch::Tensor)) {
void Communicator::multiBcast(vector<torch::Tensor> &tensor_list, const std::function<void(torch::Tensor)> &op) {
    vector<int> tensor_ranks;
    int assigned_rank = 0;
    for (unsigned i = 0; i < tensor_list.size(); i++) {
        tensor_ranks.push_back(assigned_rank%m_size);
        assigned_rank++;
    }
	for (unsigned i = 0; i < tensor_list.size(); i++) {
        torch::Tensor tensor = tensor_list[i];
        assigned_rank = tensor_ranks[i];
        if (assigned_rank == m_rank) {
            op(tensor);
        } 
        NCCLCHECK(ncclBroadcast(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), ncclFloat, assigned_rank, m_nccl_comm, m_stream)); 
    }
}

