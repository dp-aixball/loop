expName=ht_prosody_2l
#python train.py --expName $expName --data /media/zhyi/RAID/Corpus/TTS/ht_prosody --noise 4 --seq-len 30 --epochs 45 --batch-size 64 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu 1
#python train.py --expName ${expName}_noise_2 --data /media/zhyi/RAID/Corpus/TTS/ht_prosody --checkpoint checkpoints/${expName}/bestmodel.pth --noise 2 --seq-len 1000 --epochs 45 --batch-size 64 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu 1
#python train.py --expName ${expName}_noise_1 --data /media/zhyi/RAID/Corpus/TTS/ht_prosody --checkpoint checkpoints/${expName}_noise_2/bestmodel.pth --noise 1 --seq-len 1000 --epochs 45 --batch-size 64 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu 1
#python train.py --expName ${expName}_noise_0 --data /media/zhyi/RAID/Corpus/TTS/ht_prosody --checkpoint checkpoints/${expName}_noise_1/bestmodel.pth --noise 0 --seq-len 1000 --epochs 45 --batch-size 64 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu 1


#python train.py --expName ${expName}_noise_1 --data /media/zhyi/RAID/Corpus/TTS/ht_prosody --checkpoint checkpoints/${expName}_noise_1/bestmodel.pth --noise 1 --seq-len 1000 --epochs 45 --batch-size 64 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu 1
python train.py --expName ${expName}_noise_0 --data /media/zhyi/RAID/Corpus/TTS/ht_prosody --checkpoint checkpoints/${expName}_noise_1/bestmodel.pth --noise 0 --seq-len 1000 --epochs 45 --batch-size 64 --nspk 20  --attention-alignment 0.05 --vocabulary-size 210 --gpu 1
