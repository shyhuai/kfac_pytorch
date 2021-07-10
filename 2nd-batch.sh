#nworkers=64 kfac_name=inverse_reduce_layerwise ./batch.sh
#nworkers=64 kfac_name=inverse_opt ./batch.sh

rdma=0 nworkers=64 kfac_name=inverse_mpd ./batch.sh
rdma=0 nworkers=64 kfac_name=inverse_opt ./batch.sh
rdma=0 nworkers=64 kfac_name=inverse_reduce_layerwise ./batch.sh
rdma=0 nworkers=64 kfac_name=inverse_reduce_merge ./batch.sh
rdma=0 nworkers=64 kfac_name=inverse_reduce_blockpartition_opt ./batch.sh
rdma=0 nworkers=64 kfac_name=inverse_reduce_blockpartition_opt_mgwfbp ./batch.sh
