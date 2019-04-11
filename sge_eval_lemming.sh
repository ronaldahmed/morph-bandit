qsub -cwd -l mem_free=20G,act_mem_free=20G,h_vmem=30G -p -50 \
-o baselines-t2/lemming/models/eval \
-e baselines-t2/lemming/models/ev-err \
eval_lemming.sh
