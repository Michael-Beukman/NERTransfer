cd ../v10
rm eval.sh
ln -s ../v20/eval.sh .
rm evaluate_ner.py
ln -s ../../utils/evaluate_ner.py .

# For all models
for i in models/*_50; do 
    # Get names, and some more info
    N=`basename $i`; 
    echo $N
    X="${N##*v2_}"
    X="${X%%_finetune*}"
    K=0
    # For each language, 
    for lang in yor hau kin lug pcm wol swa ibo luo; do
        echo $lang;
        # evaluate
        bash eval.sh zero_shot_$N"_$lang" $i $lang &
        
        # this runs 4 in parallel
        K=$((K+1))
        if [ $((K % 4)) = 0  ]; then
           wait
        fi
    done
    wait
done
