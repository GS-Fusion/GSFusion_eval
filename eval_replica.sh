#!/bin/bash

echo "Evaluate Replica..."
echo "Start rendering..."
python render.py -m ~/Projects/GSFusion/output_office0 --dataset_type replica --iteration 2000 --data_device cuda &
PID1=$!        
python render.py -m ~/Projects/GSFusion/output_office1 --dataset_type replica --iteration 2000 --data_device cuda &
PID2=$!
python render.py -m ~/Projects/GSFusion/output_office2 --dataset_type replica --iteration 2000 --data_device cuda &
PID3=$!
python render.py -m ~/Projects/GSFusion/output_office3 --dataset_type replica --iteration 2000 --data_device cuda &
PID4=$!
python render.py -m ~/Projects/GSFusion/output_office4 --dataset_type replica --iteration 2000 --data_device cuda &
PID5=$!
python render.py -m ~/Projects/GSFusion/output_room0 --dataset_type replica --iteration 2000 --data_device cuda &
PID6=$!
python render.py -m ~/Projects/GSFusion/output_room1 --dataset_type replica --iteration 2000 --data_device cuda &
PID7=$!
python render.py -m ~/Projects/GSFusion/output_room2 --dataset_type replica --iteration 2000 --data_device cuda &
PID8=$!

echo "Start evaluating..."
wait $PID1
python eval.py --data ~/Projects/GSFusion/output_office0/train/ours_2000 --no-eval-depth
wait $PID2
python eval.py --data ~/Projects/GSFusion/output_office1/train/ours_2000 --no-eval-depth
wait $PID3
python eval.py --data ~/Projects/GSFusion/output_office2/train/ours_2000 --no-eval-depth
wait $PID4
python eval.py --data ~/Projects/GSFusion/output_office3/train/ours_2000 --no-eval-depth
wait $PID5
python eval.py --data ~/Projects/GSFusion/output_office4/train/ours_2000 --no-eval-depth
wait $PID6
python eval.py --data ~/Projects/GSFusion/output_room0/train/ours_2000 --no-eval-depth
wait $PID7
python eval.py --data ~/Projects/GSFusion/output_room1/train/ours_2000 --no-eval-depth
wait $PID8
python eval.py --data ~/Projects/GSFusion/output_room2/train/ours_2000 --no-eval-depth
