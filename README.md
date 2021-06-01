# WSD-Z-reweighting
Code for Rare and zero-shot word sense disambiguation using Z-reweighting

Envs:
1. for bert-base

   python 3.7.6

   torch 1.2.0
   
   transformer 4.1.1

2. for bert-large

   python 3.8.8
   
   torch 1.7.1
   
   transformer 3.4.4
   
   pytorch_transformers 1.1.0 (for replicating original biencoder version)


How to run code

Training Z-reweighting
```
CUDA_VISIBLE_DEVICES=2,3 python biencoder_40_Z_reweighting.py --data-path /home/ysuay/codes/LMMS/external/wsd_eval/WSD_Evaluation_Framework --ckpt bert_large_v11_r1 --encoder-name bert-base --multigpu
```

Training LDAW
```
CUDA_VISIBLE_DEVICES=0,1 python biencoder_40_DRW.py --data-path /home/ysuay/codes/LMMS/external/wsd_eval/WSD_Evaluation_Framework --ckpt bert_large_DRW_r1 --encoder-name bert-large --multigpu
```

Training B-reweighting
```
CUDA_VISIBLE_DEVICES=2,3 python biencoder_40_RW_IN_new.py --data-path /home/ysuay/codes/LMMS/external/wsd_eval/WSD_Evaluation_Framework --ckpt bert_large_reweight_sense_new_r1 --encoder-name bert-large --multigpu
```

Training B-resampling
```
CUDA_VISIBLE_DEVICES=0,1 python biencoder_40_even.py --data-path /home/ysuay/codes/LMMS/external/wsd_eval/WSD_Evaluation_Framework --ckpt bert_large_resample_even_r1 --encoder-name bert-large --multigpu
```

Evaluation
eg. for senseval2, the split can also be senseval3, semeval2013, semeval2015, ALL
```
CUDA_VISIBLE_DEVICES=0,1 ~/tools/envpy37/bin/python biencoder_40_Z_reweighting.py --data-path /home/ysuay/codes/LMMS/external/wsd_eval/WSD_Evaluation_Framework --ckpt ../wsd-baseline/resample_even --multigpu --eval --split senseval2
```
