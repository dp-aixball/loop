expName=ht_prosody
gpuid=0
epochs=45
data=/media/zhyi/RAID/Corpus/TTS/${expName}
#python train.py --expName ${expName}_l0050 --data $data --noise 0 --seq-len 50 --epochs $epochs --batch-size 64 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu $gpuid
#python train.py --expName ${expName}_l1000 --data $data --checkpoint checkpoints/${expName}_l0050/bestmodel.pth --noise 2 --seq-len 1000 --epochs $epochs --batch-size 64 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu $gpuid --max-seq-len 1000
#python train.py --expName ${expName}_l2000 --data $data --checkpoint checkpoints/${expName}_l1000/bestmodel.pth --noise 2 --seq-len 2000 --epochs $epochs --batch-size 32 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu $gpuid --max-seq-len 2000
python train.py --expName ${expName}_l2000_n1 --data $data --checkpoint checkpoints/${expName}_l2000/bestmodel.pth --noise 1 --seq-len 2000 --epochs $epochs --batch-size 32 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu $gpuid --max-seq-len 2000
