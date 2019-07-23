#!/bin/bash

exp=$1

batch="data/tbnames-thesis"



TGT=$HOME/personal_work_ms/thesis-files
loss="mle"

mkdir -p $TGT



for tb in $(cut -f 2 -d " " $batch); do
	echo $tb

    
    outdir=models-segm/$tb
    lang_name=${tb:0:2}
    input_anlz_model=""

    op_ep=$(tail -1 models-segm/$tb/log.out | cut -f 1)
    input_model=models-segm/$tb/segm_$op_ep.pth
    emb_file=""

    mkdir -p $TGT/$outdir
    mkdir -p $TGT/models-anlz/$tb/
	
    cp $input_model $TGT/$input_model
    cp models-segm/$tb/log.out $TGT/models-segm/$tb/log.out


    if [ ${exp:0:2} == "l1" ]; then
        emb_file=models-segm/$tb/emb.pth
    elif [ ${exp:0:3} == "ml1" ]; then
        if [ $lang_name == "es" ]; then
            emb_file=l1-multi-emb/cs-es/cs-es/vectors-es.pth
        else
            emb_file=l1-multi-emb/"$lang_name"-es/"$lang_name"-es/vectors-"$lang_name".pth
        fi
        mkdir -p $TGT/l1-multi-emb/"$lang_name"-es/"$lang_name"-es/
    fi
    cp $emb_file $TGT/$emb_file

    if [ $exp == "l1-a1" ]; then
        op_ep_anl=$(tail -1 models-anlz/$tb/log.out | cut -f 1)
        input_anlz_model=models-anlz/$tb/anlz_"$op_ep_anl".pth
    elif [ $exp == "l1-a2" ]; then
        op_ep_anl=$(tail -1 models-anlz/$tb/log-l1a2.out | cut -f 1)
        input_anlz_model=models-anlz/$tb/anlz_fine-seq_"$op_ep_anl".pth
        # op_ep_anl=$(tail -1 models-anlz/$tb/log-$exp.out | cut -f 1)
        # input_anlz_model=models-anlz/$tb/"$exp"_"$op_ep_anl".pth
    else
        op_ep_anl=$(tail -1 models-anlz/$tb/log-$exp.out | cut -f 1)
        input_anlz_model=models-anlz/$tb/"$exp"_"$op_ep_anl".pth
    fi

    cp models-anlz/$tb/log.out $TGT/models-anlz/$tb/log.out
    cp $input_anlz_model $TGT/$input_anlz_model



done