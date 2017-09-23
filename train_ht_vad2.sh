expName=ht_vad2
gpuid=1
epochs=100
data=/media/zhyi/RAID/Corpus/TTS/${expName}
python train.py --expName ${expName}_tr1 --data $data --noise 4 --seq-len 400  --max-seq-len 400 --epochs $epochs --batch-size 64 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu $gpuid
python train.py --expName ${expName}_tr2 --data $data --checkpoint checkpoints/${expName}_tr2/bestmodel.pth --noise 2 --seq-len 1000 --epochs $epochs --batch-size 64 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu $gpuid --max-seq-len 1000
python train.py --expName ${expName}_tr3 --data $data --checkpoint checkpoints/${expName}_tr2/bestmodel.pth --noise 2 --seq-len 2000 --epochs $epochs --batch-size 32 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu $gpuid --max-seq-len 2000
python train.py --expName ${expName}_tr4 --data $data --checkpoint checkpoints/${expName}_tr3/bestmodel.pth --noise 1 --seq-len 2000 --epochs $epochs --batch-size 32 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu $gpuid --max-seq-len 2000
