python main.py --data_folder  ../../data/taobao-100k/100k_unknown_cate/ --embedding_dim 500 --hidden_size 500 --num_layers 2 --num_heads 2 --lr 0.001 --window_size 8 --test_observed 5 --n_epochs 20 --shared_embedding 1 --position_embedding 1 --batch_size 100 --optimizer_type Adam --save_model --context item_subspace --kernel_type lin-5 