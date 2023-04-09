torchrun \
    --nnodes 1 \
    --nproc-per-node 4 \
    main.py eval mlp --num-models 4
