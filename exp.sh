python main.py --epoch 60 --white_bkgd  --things 'drums' --cuda 0 --Batch_size 512 --factor 4 --co_samples 32 --re_samples 32
python main.py --epoch 60 --white_bkgd  --things 'lego' --cuda 1 --Batch_size 512 --factor 4 --co_samples 32 --re_samples 32
python main.py --epoch 60 --warm_up 8 --anneal 45 --white_bkgd  --things 'lego' --cuda 2 --Batch_size 512 --factor 2 --co_samples 64 --re_samples 64 
python main.py --epoch 60 --warm_up 8 --anneal 45 --white_bkgd  --things 'drums' --cuda 3 --Batch_size 512 --factor 2 --co_samples 64 --re_samples 64 