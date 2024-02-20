nbpe=2000
bpemode=unigram
mkdir -p ${bpemode}
dict=${bpemode}/${bpemode}${nbpe}_units.txt
bpemodel=${bpemode}/${bpemode}${nbpe}
echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
#python spm_train.py --input=input_kor_temp.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
python spm_encode.py --model=${bpemodel}.model --output_format=piece < input_test_kor.txt | tr " " '\n' | LC_COLLATE=ko_KR.UTF-8 sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
